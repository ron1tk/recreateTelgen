import torch
import torch.nn.functional as F

from models.utils import MLP
from models.hetero_conv import HeteroConv
from models.hetero_gnn import strseq2rank, get_conv_layer


class TelgenTripartiteHeteroGNN(torch.nn.Module):
    """
    TELGEN-style double-loop tripartite hetero GNN.

    Mapping:
      - outer loops K  ~ IPM/Newton iterations
      - inner layers J ~ learned computation within one outer step

    We produce one prediction per OUTER loop so the existing trainer can keep
    treating columns as optimization steps.
    """

    def __init__(self,
                 conv,
                 in_shape,
                 pe_dim,
                 hid_dim,
                 num_outer_layers,
                 num_inner_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 dropout,
                 use_norm,
                 use_res,
                 in_place=True,
                 conv_sequence='parallel'):
        super().__init__()

        self.dropout = dropout
        self.use_res = use_res
        self.num_outer_layers = num_outer_layers
        self.num_inner_layers = num_inner_layers

        # same encoder logic as the current tripartite model so input dims stay compatible
        if pe_dim > 0:
            self.pe_encoder = torch.nn.ModuleDict({
                'vals': MLP([pe_dim, hid_dim, hid_dim]),
                'cons': MLP([pe_dim, hid_dim, hid_dim]),
                'obj': MLP([pe_dim, hid_dim, hid_dim]),
            })
            in_emb_dim = hid_dim
        else:
            self.pe_encoder = None
            in_emb_dim = 2 * hid_dim

        self.encoder = torch.nn.ModuleDict({
            'vals': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
            'cons': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
            'obj': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
        })

        c2v, v2c, v2o, o2v, c2o, o2c = strseq2rank(conv_sequence)
        get_conv = get_conv_layer(conv, 2 * hid_dim, hid_dim, num_mlp_layers, use_norm, in_place)

        # J shared inner layers: reused inside every outer loop
        self.inner_gcns = torch.nn.ModuleList()
        for _ in range(num_inner_layers):
            self.inner_gcns.append(
                HeteroConv({
                    ('cons', 'to', 'vals'): (get_conv(), c2v),
                    ('vals', 'to', 'cons'): (get_conv(), v2c),
                    ('vals', 'to', 'obj'): (get_conv(), v2o),
                    ('obj', 'to', 'vals'): (get_conv(), o2v),
                    ('cons', 'to', 'obj'): (get_conv(), c2o),
                    ('obj', 'to', 'cons'): (get_conv(), o2c),
                }, aggr='cat')
            )

        # one shared readout, applied after each OUTER loop
        self.pred_vals = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1])
        self.pred_cons = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1])

    def _encode_inputs(self, data):
        x_dict = {}
        for k in ['cons', 'vals', 'obj']:
            x_emb = self.encoder[k](data[k].x)

            if self.pe_encoder is not None and hasattr(data[k], 'laplacian_eigenvector_pe'):
                pe = data[k].laplacian_eigenvector_pe
                pe_emb = 0.5 * (
                    self.pe_encoder[k](pe) +
                    self.pe_encoder[k](-pe)
                )
                x_emb = torch.cat([x_emb, pe_emb], dim=1)

            x_dict[k] = x_emb
        return x_dict

    def _apply_inner_block(self, x_dict, edge_index_dict, edge_attr_dict):
        """
        Run the shared J-layer inner network once.
        """
        h = x_dict
        for j in range(self.num_inner_layers):
            h_prev = h
            h_new = self.inner_gcns[j](h, edge_index_dict, edge_attr_dict)
            keys = h_new.keys()

            if self.use_res:
                h = {k: (F.relu(h_new[k]) + h_prev[k]) / 2 for k in keys}
            else:
                h = {k: F.relu(h_new[k]) for k in keys}

            h = {k: F.dropout(h[k], p=self.dropout, training=self.training) for k in keys}
        return h

    def forward(self, data):
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict

        # initial encoded state
        x_dict = self._encode_inputs(data)

        vals_per_outer = []
        cons_per_outer = []

        # K outer loops: one output per outer loop
        for _ in range(self.num_outer_layers):
            x_dict = self._apply_inner_block(x_dict, edge_index_dict, edge_attr_dict)
            vals_per_outer.append(x_dict['vals'])
            cons_per_outer.append(x_dict['cons'])

        vals = self.pred_vals(torch.stack(vals_per_outer, dim=0)).squeeze(-1).transpose(0, 1)
        cons = self.pred_cons(torch.stack(cons_per_outer, dim=0)).squeeze(-1).transpose(0, 1)

        # TE path split ratios should be non-negative
        vals = F.relu(vals)

        return vals, cons