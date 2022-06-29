# Copyright 2021 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_sparse import SparseTensor
from torch_geometric.nn.dense.linear import Linear

from typing import List, Tuple, Union, Dict

from models.conv_sample import BEANConvSample
from models.sampler import BEANAdjacency, BipartiteNeighborSampler, EdgeLoader
from utils.sparse_combine import xe_split3

from tqdm import tqdm


def make_tuple(x: Union[int, Tuple[int, int, int], Tuple[int, int]], repeat: int = 3):
    if isinstance(x, int):
        if repeat == 2:
            return (x, x)
        else:
            return (x, x, x)
    else:
        return x


def apply_relu_dropout(x: Tensor, dropout_prob: float, training: bool) -> Tensor:
    x = F.relu(x)
    if dropout_prob > 0.0:
        x = F.dropout(x, p=dropout_prob, training=training)
    return x


class GraphBEANSampled(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int, int]],
        hidden_channels: Union[int, Tuple[int, int, int]] = 32,
        latent_channels: Union[int, Tuple[int, int]] = 64,
        edge_pred_latent: int = 64,
        n_layers_encoder: int = 4,
        n_layers_decoder: int = 4,
        n_layers_mlp: int = 4,
        dropout_prob: float = 0.0,
    ):

        super().__init__()

        self.in_channels = make_tuple(in_channels)
        self.hidden_channels = make_tuple(hidden_channels)
        self.latent_channels = make_tuple(latent_channels, 2)
        self.edge_pred_latent = edge_pred_latent
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.n_layers_mlp = n_layers_mlp
        self.dropout_prob = dropout_prob

        self.create_encoder()
        self.create_feature_decoder()
        self.create_structure_decoder()

    def create_encoder(self):
        self.encoder_convs = nn.ModuleList()
        for i in range(self.n_layers_encoder):
            if i == 0:
                in_channels = self.in_channels
                out_channels = self.hidden_channels
            elif i == self.n_layers_encoder - 1:
                in_channels = self.hidden_channels
                out_channels = self.latent_channels
            else:
                in_channels = self.hidden_channels
                out_channels = self.hidden_channels

            if i == self.n_layers_encoder - 1:
                self.encoder_convs.append(
                    BEANConvSample(in_channels, out_channels, node_self_loop=False)
                )
            else:
                self.encoder_convs.append(
                    BEANConvSample(in_channels, out_channels, node_self_loop=True)
                )

    def create_feature_decoder(self):
        self.decoder_convs = nn.ModuleList()
        for i in range(self.n_layers_decoder):
            if i == 0:
                in_channels = self.latent_channels
                out_channels = self.hidden_channels
            elif i == self.n_layers_decoder - 1:
                in_channels = self.hidden_channels
                out_channels = self.in_channels
            else:
                in_channels = self.hidden_channels
                out_channels = self.hidden_channels

            self.decoder_convs.append(BEANConvSample(in_channels, out_channels))

    def create_structure_decoder(self):
        self.u_mlp_layers = nn.ModuleList()
        self.v_mlp_layers = nn.ModuleList()

        for i in range(self.n_layers_mlp):
            if i == 0:
                in_channels = self.latent_channels
            else:
                in_channels = (self.edge_pred_latent, self.edge_pred_latent)
            out_channels = self.edge_pred_latent

            self.u_mlp_layers.append(Linear(in_channels[0], out_channels))

            self.v_mlp_layers.append(Linear(in_channels[1], out_channels))

    def forward(
        self,
        xu: Tensor,
        xv: Tensor,
        xe: Tensor,
        bean_adjs: List[BEANAdjacency],
        e_flags: List[SparseTensor],
        edge_pred_samples: SparseTensor,
    ) -> Dict[str, Tensor]:

        assert self.n_layers_encoder + self.n_layers_decoder == len(bean_adjs)

        # encoder
        for i, conv in enumerate(self.encoder_convs):
            badj = bean_adjs[i]
            e_flag = e_flags[i]

            # target size
            n_ut = badj.adj_v2u.size[0]
            n_vt = badj.adj_u2v.size[1]

            # get xut and xvt
            xus, xut = xu, xu[:n_ut]  #  target nodes are always placed first
            xvs, xvt = xv, xv[:n_vt]

            # get xe
            xe_e, xe_v2u, xe_u2v = xe_split3(xe, e_flag.storage.value())

            # do convolution
            xu, xv, xe = conv(
                xu=(xus, xut), xv=(xvs, xvt), adj=badj, xe=(xe_e, xe_v2u, xe_u2v)
            )

            if i != self.n_layers_encoder - 1:
                xu = apply_relu_dropout(xu, self.dropout_prob, self.training)
                xv = apply_relu_dropout(xv, self.dropout_prob, self.training)
                xe = apply_relu_dropout(xe, self.dropout_prob, self.training)

        # extract latent vars (only target nodes)
        last_badj = bean_adjs[-1]
        n_u_target = last_badj.adj_v2u.size[0]
        n_v_target = last_badj.adj_u2v.size[1]
        # get latent vars
        zu, zv = xu[:n_u_target], xv[:n_v_target]

        # feature decoder
        for i, conv in enumerate(self.decoder_convs):

            badj = bean_adjs[self.n_layers_encoder + i]
            e_flag = e_flags[self.n_layers_encoder + i]

            # target size
            n_ut = badj.adj_v2u.size[0]
            n_vt = badj.adj_u2v.size[1]

            # get xut and xvt
            xus, xut = xu, xu[:n_ut]  #  target nodes are always placed first
            xvs, xvt = xv, xv[:n_vt]

            # get xe
            if xe is not None:
                xe_e, xe_v2u, xe_u2v = xe_split3(xe, e_flag.storage.value())
            else:
                xe_e, xe_v2u, xe_u2v = None, None, None

            # do convolution
            xu, xv, xe = conv(
                xu=(xus, xut), xv=(xvs, xvt), adj=badj, xe=(xe_e, xe_v2u, xe_u2v)
            )

            if i != self.n_layers_decoder - 1:
                xu = apply_relu_dropout(xu, self.dropout_prob, self.training)
                xv = apply_relu_dropout(xv, self.dropout_prob, self.training)
                xe = apply_relu_dropout(xe, self.dropout_prob, self.training)

        # structure decoder
        zu2, zv2 = zu, zv
        for i, layer in enumerate(self.u_mlp_layers):
            zu2 = layer(zu2)
            if i != self.n_layers_mlp - 1:
                zu2 = apply_relu_dropout(zu2, self.dropout_prob, self.training)

        for i, layer in enumerate(self.v_mlp_layers):
            zv2 = layer(zv2)
            if i != self.n_layers_mlp - 1:
                zv2 = apply_relu_dropout(zv2, self.dropout_prob, self.training)

        zu2_edge = zu2[edge_pred_samples.storage.row()]
        zv2_edge = zv2[edge_pred_samples.storage.col()]

        eprob = torch.sigmoid(torch.sum(zu2_edge * zv2_edge, dim=1))

        # collect results
        result = {"xu": xu, "xv": xv, "xe": xe, "zu": zu, "zv": zv, "eprob": eprob}

        return result

    def apply_conv(self, conv, dir_adj, xu_all, xv_all, xe_all, device):
        xu = xu_all[dir_adj.u_id].to(device)
        xv = xv_all[dir_adj.v_id].to(device)
        xe = xe_all[dir_adj.e_id].to(device) if xe_all is not None else None
        adj = dir_adj.adj.to(device)

        out = conv((xu, xv), adj, xe)

        return out

    def inference(
        self,
        xu_all: Tensor,
        xv_all: Tensor,
        xe_all: Tensor,
        adj_all: SparseTensor,
        edge_pred_samples: SparseTensor,
        batch_sizes: Tuple[int, int, int],
        device,
        progress_bar: bool = True,
        **kwargs,
    ) -> Dict[str, Tensor]:

        kwargs["shuffle"] = False
        u_loader = BipartiteNeighborSampler(
            adj_all,
            n_layers=1,
            base="u",
            batch_size=batch_sizes[0],
            n_other_node=1,
            num_neighbors_u=-1,
            num_neighbors_v=1,
            **kwargs,
        )
        v_loader = BipartiteNeighborSampler(
            adj_all,
            n_layers=1,
            base="v",
            batch_size=batch_sizes[1],
            n_other_node=1,
            num_neighbors_u=1,
            num_neighbors_v=-1,
            **kwargs,
        )
        e_loader = EdgeLoader(adj_all, batch_size=batch_sizes[2], **kwargs)

        u_mlp_loader = torch.utils.data.DataLoader(
            torch.arange(xu_all.shape[0]), batch_size=batch_sizes[0], **kwargs
        )
        v_mlp_loader = torch.utils.data.DataLoader(
            torch.arange(xv_all.shape[0]), batch_size=batch_sizes[1], **kwargs
        )

        epred_loader = torch.utils.data.DataLoader(
            torch.arange(edge_pred_samples.nnz()), batch_size=batch_sizes[2], **kwargs
        )

        total_iter = (
            (len(u_loader) + len(v_loader))
            * (self.n_layers_encoder + self.n_layers_decoder)
            + len(e_loader) * (self.n_layers_encoder + self.n_layers_decoder - 1)
            + (len(u_mlp_loader) + len(v_mlp_loader)) * self.n_layers_mlp
            + len(epred_loader)
        )
        if progress_bar:
            pbar = tqdm(total=total_iter, leave=False)
            pbar.set_description(f"Evaluation")

        # encoder
        for i, conv in enumerate(self.encoder_convs):

            ## next u nodes
            xu_list = []
            for _, _, adjacency, _ in u_loader:
                out = self.apply_conv(
                    conv.v2u_conv, adjacency.adj_v2u, xu_all, xv_all, xe_all, device
                )
                if i != self.n_layers_encoder - 1:
                    out = F.relu(out)
                xu_list.append(out.cpu())
                if progress_bar:
                    pbar.update(1)
            xu_all_next = torch.cat(xu_list, dim=0)

            ## next v nodes
            xv_list = []
            for _, _, adjacency, _ in v_loader:
                out = self.apply_conv(
                    conv.u2v_conv, adjacency.adj_u2v, xu_all, xv_all, xe_all, device
                )
                if i != self.n_layers_encoder - 1:
                    out = F.relu(out)
                xv_list.append(out.cpu())
                if progress_bar:
                    pbar.update(1)
            xv_all_next = torch.cat(xv_list, dim=0)

            ## next edge
            if i != self.n_layers_encoder - 1:
                xe_list = []
                for adj_e in e_loader:
                    out = self.apply_conv(
                        conv.e_conv, adj_e, xu_all, xv_all, xe_all, device
                    )
                    out = F.relu(out)
                    xe_list.append(out.cpu())
                    if progress_bar:
                        pbar.update(1)
                xe_all_next = torch.cat(xe_list, dim=0)
            else:
                xe_all_next = None

            xu_all = xu_all_next
            xv_all = xv_all_next
            xe_all = xe_all_next

        # get latent vars
        zu_all, zv_all = xu_all, xv_all

        # feature decoder
        for i, conv in enumerate(self.decoder_convs):

            ## next u nodes
            xu_list = []
            for _, _, adjacency, _ in u_loader:
                out = self.apply_conv(
                    conv.v2u_conv, adjacency.adj_v2u, xu_all, xv_all, xe_all, device
                )
                if i != self.n_layers_decoder - 1:
                    out = F.relu(out)
                xu_list.append(out.cpu())
                if progress_bar:
                    pbar.update(1)
            xu_all_next = torch.cat(xu_list, dim=0)

            ## next v nodes
            xv_list = []
            for _, _, adjacency, _ in v_loader:
                out = self.apply_conv(
                    conv.u2v_conv, adjacency.adj_u2v, xu_all, xv_all, xe_all, device
                )
                if i != self.n_layers_decoder - 1:
                    out = F.relu(out)
                xv_list.append(out.cpu())
                if progress_bar:
                    pbar.update(1)
            xv_all_next = torch.cat(xv_list, dim=0)

            ## next edge
            xe_list = []
            for adj_e in e_loader:
                out = self.apply_conv(
                    conv.e_conv, adj_e, xu_all, xv_all, xe_all, device
                )
                if i != self.n_layers_decoder - 1:
                    out = F.relu(out)
                xe_list.append(out.cpu())
                if progress_bar:
                    pbar.update(1)
            xe_all_next = torch.cat(xe_list, dim=0)

            xu_all = xu_all_next
            xv_all = xv_all_next
            xe_all = xe_all_next

        # structure decoder
        zu2_all, zv2_all = zu_all, zv_all
        for i, layer in enumerate(self.u_mlp_layers):
            zu2_list = []
            for batch in u_mlp_loader:
                out = layer(zu2_all[batch].to(device))
                if i != self.n_layers_mlp - 1:
                    out = F.relu(out)
                zu2_list.append(out.cpu())
                if progress_bar:
                    pbar.update(1)
            zu2_all = torch.cat(zu2_list, dim=0)

        for i, layer in enumerate(self.v_mlp_layers):
            zv2_list = []
            for batch in v_mlp_loader:
                out = layer(zv2_all[batch].to(device))
                if i != self.n_layers_mlp - 1:
                    out = F.relu(out)
                zv2_list.append(out.cpu())
                if progress_bar:
                    pbar.update(1)
            zv2_all = torch.cat(zv2_list, dim=0)

        eprob_list = []
        for batch in epred_loader:
            zu2_edge = zu2_all[edge_pred_samples.storage.row()[batch]].to(device)
            zv2_edge = zv2_all[edge_pred_samples.storage.col()[batch]].to(device)
            out = torch.sigmoid(torch.sum(zu2_edge * zv2_edge, dim=1))
            eprob_list.append(out.cpu())
            if progress_bar:
                pbar.update(1)
        eprob_all = torch.cat(eprob_list, dim=0)

        # collect results
        result = {
            "xu": xu_all,
            "xv": xv_all,
            "xe": xe_all,
            "zu": zu_all,
            "zv": zv_all,
            "eprob": eprob_all,
        }

        if progress_bar:
            pbar.close()

        return result
