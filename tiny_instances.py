import os
import gzip
import pickle
import torch
import numpy as np
from functools import partial

from generate_instances import (
    generate_setcover,
    Graph,
    generate_indset,
    generate_cauctions,
    generate_capacited_facility_location
)

# -----------------------------
# Config
# -----------------------------
root = os.environ.get("DATAPATH", "./instance_data")
raw_dir = os.path.join(root, "raw")
os.makedirs(raw_dir, exist_ok=True)

rng = np.random.RandomState(42)
bounds = (0., 1.)

# -----------------------------
# Helper to save a single instance
# -----------------------------
def save_instance(A, b, c, filename):
    data = [(torch.from_numpy(A).float(),
             torch.from_numpy(b).float(),
             torch.from_numpy(c).float())]
    with gzip.open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved: {filename}")

# -----------------------------
# 1. Tiny Setcover
# -----------------------------
density = 0.5
nrows_l, nrows_u = 5, 6
ncols_l, ncols_u = 5, 6

A, b, c = generate_setcover(
    nrows_l=nrows_l, nrows_u=nrows_u,
    ncols_l=ncols_l, ncols_u=ncols_u,
    density=density, rng=rng
)
save_instance(A, b, c, os.path.join(raw_dir, "instance_0.pkl.gz"))

# -----------------------------
# 2. Tiny Indset
# -----------------------------
nnodes = 5
graph = Graph.erdos_renyi(number_of_nodes=nnodes, edge_probability=0.4, random=rng)
A, b, c = generate_indset(graph, nnodes)
save_instance(A, b, c, os.path.join(raw_dir, "instance_1.pkl.gz"))

# -----------------------------
# 3. Tiny Combinatorial Auction
# -----------------------------
n_items = 5
n_bids = 5
A, b, c = generate_cauctions(n_items=n_items, n_bids=n_bids, rng=rng, min_value=1, max_value=10)
save_instance(A, b, c, os.path.join(raw_dir, "instance_2.pkl.gz"))

# -----------------------------
# 4. Tiny Facility Location
# -----------------------------
n_customers = 3
n_facilities = 3
ratio = 1.5
A, b, c = generate_capacited_facility_location(n_customers, n_facilities, ratio, rng)
save_instance(A, b, c, os.path.join(raw_dir, "instance_3.pkl.gz"))

print("All tiny instances generated!")
