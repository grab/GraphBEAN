# Copyright 2021 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

from typing import Optional, Tuple
import torch

from torch import Tensor
import torch.nn as nn
from torch_sparse import SparseTensor, matmul
from torch_scatter import scatter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear

from torch_geometric.typing import PairTensor, OptTensor


class BEANConv(MessagePassing):
    def __init__(
        self,
        in_channels: Tuple[int, int, Optional[int]],
        out_channels: Tuple[int, int, Optional[int]],
        node_self_loop: bool = True,
        normalize: bool = True,
        bias: bool = True,
        **kwargs
    ):

        super(BEANConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.node_self_loop = node_self_loop
        self.normalize = normalize

        self.input_has_edge_channel = len(in_channels) == 3
        self.output_has_edge_channel = len(out_channels) == 3

        if self.input_has_edge_channel:
            if self.node_self_loop:
                self.in_channels_u = (
                    in_channels[0] + 2 * in_channels[1] + 2 * in_channels[2]
                )
                self.in_channels_v = (
                    2 * in_channels[0] + in_channels[1] + 2 * in_channels[2]
                )
            else:
                self.in_channels_u = 2 * in_channels[1] + 2 * in_channels[2]
                self.in_channels_v = 2 * in_channels[0] + 2 * in_channels[2]
            self.in_channels_e = in_channels[0] + in_channels[1] + in_channels[2]
        else:
            if self.node_self_loop:
                self.in_channels_u = in_channels[0] + 2 * in_channels[1]
                self.in_channels_v = 2 * in_channels[0] + in_channels[1]
            else:
                self.in_channels_u = 2 * in_channels[1]
                self.in_channels_v = 2 * in_channels[0]
            self.in_channels_e = in_channels[0] + in_channels[1]

        self.lin_u = Linear(self.in_channels_u, out_channels[0], bias=bias)
        self.lin_v = Linear(self.in_channels_v, out_channels[1], bias=bias)
        if self.output_has_edge_channel:
            self.lin_e = Linear(self.in_channels_e, out_channels[2], bias=bias)

        if normalize:
            self.bn_u = nn.BatchNorm1d(out_channels[0])
            self.bn_v = nn.BatchNorm1d(out_channels[1])
            if self.output_has_edge_channel:
                self.bn_e = nn.BatchNorm1d(out_channels[2])

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_u.reset_parameters()
        self.lin_v.reset_parameters()
        if self.output_has_edge_channel:
            self.lin_e.reset_parameters()

    def forward(
        self, x: PairTensor, adj: SparseTensor, xe: OptTensor = None
    ) -> Tuple[PairTensor, Tensor]:
        """"""

        assert self.input_has_edge_channel == (xe is not None)

        # propagate_type: (x: PairTensor)
        (out_u, out_v), out_e = self.propagate(adj, x=x, xe=xe)

        # lin layer
        out_u = self.lin_u(out_u)
        out_v = self.lin_v(out_v)
        if self.output_has_edge_channel:
            out_e = self.lin_e(out_e)

        if self.normalize:
            out_u = self.bn_u(out_u)
            out_v = self.bn_v(out_v)
            if self.output_has_edge_channel:
                out_e = self.bn_e(out_e)

        return (out_u, out_v), out_e

    def message_and_aggregate(
        self, adj: SparseTensor, x: PairTensor, xe: OptTensor
    ) -> Tuple[PairTensor, Tensor]:

        xu, xv = x
        adj = adj.set_value(None, layout=None)

        # messages node to node
        msg_v2u_mean = matmul(adj, xv, reduce="mean")
        msg_v2u_sum = matmul(adj, xv, reduce="max")

        msg_u2v_mean = matmul(adj.t(), xu, reduce="mean")
        msg_u2v_sum = matmul(adj.t(), xu, reduce="max")

        # messages edge to node
        if xe is not None:
            msg_e2u_mean = scatter(xe, adj.storage.row(), dim=0, reduce="mean")
            msg_e2u_sum = scatter(xe, adj.storage.row(), dim=0, reduce="max")

            msg_e2v_mean = scatter(xe, adj.storage.col(), dim=0, reduce="mean")
            msg_e2v_sum = scatter(xe, adj.storage.col(), dim=0, reduce="max")

        # collect all msg (including self loop)
        msg_2e = None
        if xe is not None:
            if self.node_self_loop:
                msg_2u = torch.cat(
                    (xu, msg_v2u_mean, msg_v2u_sum, msg_e2u_mean, msg_e2u_sum), dim=1
                )
                msg_2v = torch.cat(
                    (xv, msg_u2v_mean, msg_u2v_sum, msg_e2v_mean, msg_e2v_sum), dim=1
                )
            else:
                msg_2u = torch.cat(
                    (msg_v2u_mean, msg_v2u_sum, msg_e2u_mean, msg_e2u_sum), dim=1
                )
                msg_2v = torch.cat(
                    (msg_u2v_mean, msg_u2v_sum, msg_e2v_mean, msg_e2v_sum), dim=1
                )

            if self.output_has_edge_channel:
                msg_2e = torch.cat(
                    (xe, xu[adj.storage.row()], xv[adj.storage.col()]), dim=1
                )
        else:
            if self.node_self_loop:
                msg_2u = torch.cat((xu, msg_v2u_mean, msg_v2u_sum), dim=1)
                msg_2v = torch.cat((xv, msg_u2v_mean, msg_u2v_sum), dim=1)
            else:
                msg_2u = torch.cat((msg_v2u_mean, msg_v2u_sum), dim=1)
                msg_2v = torch.cat((msg_u2v_mean, msg_u2v_sum), dim=1)

            if self.output_has_edge_channel:
                msg_2e = torch.cat(
                    (xu[adj.storage.row()], xv[adj.storage.col()]), dim=1
                )

        return (msg_2u, msg_2v), msg_2e
