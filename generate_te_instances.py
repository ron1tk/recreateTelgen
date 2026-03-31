import argparse
import gzip
import json
import os
import os.path as osp
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    import networkx as nx
except ImportError as e:
    raise SystemExit(
        "This script needs networkx. Install it in your env with:\n"
        "  pip install networkx"
    ) from e


def build_connected_bidirectional_graph(
    n_nodes: int,
    edge_prob: float,
    cap_min: float,
    cap_max: float,
    rng: np.random.RandomState,
) -> nx.DiGraph:
    """
    Build a connected undirected graph first, then convert it to a symmetric
    directed graph so every undirected link becomes two directed arcs.
    """
    if n_nodes < 2:
        raise ValueError("n_nodes must be >= 2")

    # Start with a random spanning tree to guarantee connectivity.
    perm = rng.permutation(n_nodes)
    undirected_edges = set()
    for i in range(1, n_nodes):
        u = int(perm[i])
        v = int(perm[rng.randint(0, i)])
        undirected_edges.add(tuple(sorted((u, v))))

    # Add extra undirected edges.
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            if (u, v) in undirected_edges:
                continue
            if rng.rand() < edge_prob:
                undirected_edges.add((u, v))

    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))

    for u, v in undirected_edges:
        cap_uv = float(rng.uniform(cap_min, cap_max))
        cap_vu = float(rng.uniform(cap_min, cap_max))

        # Use inverse-capacity-ish weights so shortest paths tend to prefer
        # higher-capacity links a bit, while still remaining simple.
        g.add_edge(u, v, capacity=cap_uv, weight=1.0 / max(cap_uv, 1e-6))
        g.add_edge(v, u, capacity=cap_vu, weight=1.0 / max(cap_vu, 1e-6))

    return g


def sample_demands(
    g: nx.DiGraph,
    num_demands: int,
    demand_min: float,
    demand_max: float,
    rng: np.random.RandomState,
) -> List[Tuple[int, int, float]]:
    """
    Sample ordered source-destination pairs with positive demand.
    """
    nodes = list(g.nodes())
    ordered_pairs = [(u, v) for u in nodes for v in nodes if u != v]
    rng.shuffle(ordered_pairs)

    demands = []
    for s, t in ordered_pairs:
        if len(demands) >= num_demands:
            break
        if nx.has_path(g, s, t):
            d = float(rng.uniform(demand_min, demand_max))
            demands.append((s, t, d))

    if len(demands) < num_demands:
        raise RuntimeError(
            f"Could only find {len(demands)} reachable SD pairs, "
            f"but requested {num_demands}."
        )
    return demands


def k_candidate_paths(
    g: nx.DiGraph,
    s: int,
    t: int,
    k: int,
) -> List[List[int]]:
    """
    Use k shortest simple paths under the edge 'weight' attribute.
    """
    try:
        gen = nx.shortest_simple_paths(g, s, t, weight="weight")
        paths = []
        for _ in range(k):
            try:
                paths.append(next(gen))
            except StopIteration:
                break
        return paths
    except nx.NetworkXNoPath:
        return []


def path_to_directed_edges(path: List[int]) -> List[Tuple[int, int]]:
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def te_lp_to_Abc(
    g: nx.DiGraph,
    demands: List[Tuple[int, int, float]],
    k_paths: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Build a throughput-maximization TE LP in minimization form.

    Variables:
        x_p = split ratio on candidate path p, with 0 <= x_p <= 1

    Demand constraints:
        sum_{p in P_i} x_p <= 1

    Link constraints (normalized so RHS=1):
        sum_{i} sum_{p in P_i : l in p} (d_i / C_l) * x_p <= 1

    Objective:
        maximize sum_i d_i * sum_{p in P_i} x_p
    For scipy linprog minimization, we use c_p = -d_i.
    """
    edge_list = list(g.edges())
    edge_to_idx = {e: idx for idx, e in enumerate(edge_list)}

    # Enumerate path variables.
    path_records = []
    demand_to_path_indices: List[List[int]] = [[] for _ in range(len(demands))]

    for i, (s, t, d_i) in enumerate(demands):
        paths = k_candidate_paths(g, s, t, k_paths)
        if len(paths) == 0:
            raise RuntimeError(f"No candidate paths found for demand {(s, t)}")

        for p in paths:
            var_idx = len(path_records)
            edges_on_path = path_to_directed_edges(p)
            path_records.append(
                {
                    "demand_idx": i,
                    "source": s,
                    "target": t,
                    "demand": d_i,
                    "path_nodes": p,
                    "path_edges": edges_on_path,
                }
            )
            demand_to_path_indices[i].append(var_idx)

    num_demands = len(demands)
    num_links = len(edge_list)
    num_vars = len(path_records)
    num_cons = num_demands + num_links

    A = np.zeros((num_cons, num_vars), dtype=np.float32)
    b = np.ones((num_cons,), dtype=np.float32)
    c = np.zeros((num_vars,), dtype=np.float32)

    # Demand constraints: sum path splits for one SD pair <= 1
    for i, (_, _, d_i) in enumerate(demands):
        for var_idx in demand_to_path_indices[i]:
            A[i, var_idx] = 1.0
            c[var_idx] = -float(d_i)  # minimize negative throughput

    # Link-capacity constraints, normalized so RHS=1
    # sum (d_i / C_l) x_p <= 1
    for var_idx, rec in enumerate(path_records):
        d_i = float(rec["demand"])
        for e in rec["path_edges"]:
            link_row = num_demands + edge_to_idx[e]
            cap = float(g.edges[e]["capacity"])
            A[link_row, var_idx] += d_i / max(cap, 1e-6)

    A_t = torch.from_numpy(A)
    b_t = torch.from_numpy(b)
    c_t = torch.from_numpy(c)

    metadata = {
        "num_nodes": g.number_of_nodes(),
        "num_directed_links": num_links,
        "num_demands": num_demands,
        "num_vars": num_vars,
        "edges": [
            {"u": int(u), "v": int(v), "capacity": float(g.edges[(u, v)]["capacity"])}
            for (u, v) in edge_list
        ],
        "demands": [
            {"source": int(s), "target": int(t), "demand": float(d)}
            for (s, t, d) in demands
        ],
        "paths": path_records,
    }
    return A_t, b_t, c_t, metadata


def generate_one_instance(
    n_nodes_min: int,
    n_nodes_max: int,
    edge_prob: float,
    num_demands_min: int,
    num_demands_max: int,
    demand_min: float,
    demand_max: float,
    cap_min: float,
    cap_max: float,
    k_paths: int,
    rng: np.random.RandomState,
    max_retries: int = 20,
):
    """
    Retry generation if a sampled graph/path set is degenerate.
    """
    last_err = None
    for _ in range(max_retries):
        try:
            n_nodes = int(rng.randint(n_nodes_min, n_nodes_max + 1))
            g = build_connected_bidirectional_graph(
                n_nodes=n_nodes,
                edge_prob=edge_prob,
                cap_min=cap_min,
                cap_max=cap_max,
                rng=rng,
            )
            num_demands = int(rng.randint(num_demands_min, num_demands_max + 1))
            demands = sample_demands(
                g=g,
                num_demands=num_demands,
                demand_min=demand_min,
                demand_max=demand_max,
                rng=rng,
            )
            A, b, c, meta = te_lp_to_Abc(g, demands, k_paths=k_paths)

            # Basic sanity checks
            if A.shape[0] == 0 or A.shape[1] == 0:
                raise RuntimeError("Degenerate LP with zero rows or columns")
            if torch.isnan(A).any() or torch.isnan(b).any() or torch.isnan(c).any():
                raise RuntimeError("NaN detected in generated LP")
            if (b <= 0).any():
                raise RuntimeError("RHS should be positive after normalization")

            return (A, b, c), meta
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed to generate TE instance after retries: {last_err}")


def chunked(seq, chunk_size: int):
    for i in range(0, len(seq), chunk_size):
        yield seq[i:i + chunk_size]


def main():
    parser = argparse.ArgumentParser(description="Generate TELGEN-like TE LP instances.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Dataset root dir, e.g. datasets/te_small")
    parser.add_argument("--num_instances", type=int, default=200)
    parser.add_argument("--instances_per_file", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--n_nodes_min", type=int, default=8)
    parser.add_argument("--n_nodes_max", type=int, default=14)
    parser.add_argument("--edge_prob", type=float, default=0.25)

    parser.add_argument("--num_demands_min", type=int, default=6)
    parser.add_argument("--num_demands_max", type=int, default=16)
    parser.add_argument("--demand_min", type=float, default=5.0)
    parser.add_argument("--demand_max", type=float, default=40.0)

    parser.add_argument("--cap_min", type=float, default=30.0)
    parser.add_argument("--cap_max", type=float, default=120.0)
    parser.add_argument("--k_paths", type=int, default=3)

    args = parser.parse_args()
    rng = np.random.RandomState(args.seed)

    raw_dir = osp.join(args.output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    instances = []
    metadata = []

    for idx in range(args.num_instances):
        (A, b, c), meta = generate_one_instance(
            n_nodes_min=args.n_nodes_min,
            n_nodes_max=args.n_nodes_max,
            edge_prob=args.edge_prob,
            num_demands_min=args.num_demands_min,
            num_demands_max=args.num_demands_max,
            demand_min=args.demand_min,
            demand_max=args.demand_max,
            cap_min=args.cap_min,
            cap_max=args.cap_max,
            k_paths=args.k_paths,
            rng=rng,
        )
        instances.append((A, b, c))
        metadata.append(meta)

        if (idx + 1) % 10 == 0 or (idx + 1) == args.num_instances:
            print(
                f"[{idx + 1}/{args.num_instances}] "
                f"last instance: rows={A.shape[0]}, cols={A.shape[1]}"
            )

    for file_idx, pkg in enumerate(chunked(instances, args.instances_per_file)):
        out_path = osp.join(raw_dir, f"instance_{file_idx}.pkl.gz")
        with gzip.open(out_path, "wb") as f:
            pickle.dump(pkg, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"wrote {len(pkg)} instances to {out_path}")

    # IMPORTANT:
    # do not save extra .pkl.gz files in raw/, because LPDataset counts those.
    meta_path = osp.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "num_instances": args.num_instances,
                "instances_per_file": args.instances_per_file,
                "n_nodes_min": args.n_nodes_min,
                "n_nodes_max": args.n_nodes_max,
                "edge_prob": args.edge_prob,
                "num_demands_min": args.num_demands_min,
                "num_demands_max": args.num_demands_max,
                "demand_min": args.demand_min,
                "demand_max": args.demand_max,
                "cap_min": args.cap_min,
                "cap_max": args.cap_max,
                "k_paths": args.k_paths,
                "examples": metadata[:5],  # only a few examples to keep file small
            },
            f,
            indent=2,
        )
    print(f"wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()