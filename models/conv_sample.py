# Copyright 2021 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

from typing import List, Optional, Tuple
import torch

from torch import Tensor
import torch.nn as nn
from torch_sparse import SparseTensor, matmul
from torch_scatter import scatter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear

from torch_geometric.typing import PairTensor, OptTensor

from models.sampler import BEANAdjacency


class BEANConvSample(torch.nn.Module):
    
    def __init__(self, in_channels: Tuple[int, int, Optional[int]],
                 out_channels: Tuple[int, int, Optional[int]], 
                 node_self_loop: bool = True,
                 normalize: bool = True,
                 bias: bool = True, **kwargs):  

        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.node_self_loop = node_self_loop
        self.normalize = normalize

        self.input_has_edge_channel = (len(in_channels) == 3)
        self.output_has_edge_channel = (len(out_channels) == 3)

        self.v2u_conv = BEANConvNode(in_channels, out_channels[0], flow='v->u',
                                     node_self_loop=node_self_loop, normalize=normalize, 
                                     bias=bias, **kwargs)

        self.u2v_conv = BEANConvNode(in_channels, out_channels[1], flow='u->v',
                                     node_self_loop=node_self_loop, normalize=normalize, 
                                     bias=bias, **kwargs)

        if self.output_has_edge_channel:
            self.e_conv = BEANConvEdge(in_channels, out_channels[2], node_self_loop=node_self_loop,
                                       normalize=normalize, bias=bias, **kwargs)


    def forward(self, xu: PairTensor, xv: PairTensor, adj: BEANAdjacency, 
                xe: Optional[Tuple[Tensor, Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor, Tensor]:

        # source and target
        xus, xut = xu
        xvs, xvt = xv

        # xe
        if xe is not None:
            xe_e, xe_v2u, xe_u2v = xe

        out_u = self.v2u_conv((xut, xvs), adj.adj_v2u.adj, xe_v2u)
        out_v = self.u2v_conv((xus, xvt), adj.adj_u2v.adj, xe_u2v)

        out_e = None
        if self.output_has_edge_channel:
            out_e = self.e_conv((xut, xvt), adj.adj_e.adj, xe_e)
        
        return out_u, out_v, out_e


class BEANConvNode(MessagePassing):
    
    def __init__(self, in_channels: Tuple[int, int, Optional[int]],
                 out_channels: int, 
                 flow: str = 'v->u',
                 node_self_loop: bool = True,
                 normalize: bool = True,
                 bias: bool = True, 
                 agg: List[str] = ['mean', 'max'], 
                 **kwargs):  

        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flow = flow
        self.node_self_loop = node_self_loop
        self.normalize = normalize
        self.agg = agg

        self.input_has_edge_channel = (len(in_channels) == 3)

        n_agg = len(agg)
        # calculate in channels
        if self.input_has_edge_channel:
            if self.node_self_loop:
                if flow == 'v->u':
                    self.in_channels_all = in_channels[0] + n_agg * in_channels[1] + n_agg * in_channels[2]
                else:
                    self.in_channels_all = n_agg * in_channels[0] + in_channels[1] + n_agg * in_channels[2]
            else:
                if flow == 'v->u':
                    self.in_channels_all = n_agg * in_channels[1] + n_agg * in_channels[2]
                else:
                    self.in_channels_all = n_agg * in_channels[0] + n_agg * in_channels[2]
        else:
            if self.node_self_loop:
                if flow == 'v->u':
                    self.in_channels_all = in_channels[0] + n_agg * in_channels[1]
                else:
                    self.in_channels_all = n_agg * in_channels[0] + in_channels[1]
            else:
                if flow == 'v->u':
                    self.in_channels_all = n_agg * in_channels[1]
                else:
                    self.in_channels_all = n_agg * in_channels[0]

        self.lin = Linear(self.in_channels_all, out_channels, bias=bias)

        if normalize:
            self.bn = nn.BatchNorm1d(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
       
    def forward(self, x: PairTensor, adj: SparseTensor, 
                xe: OptTensor = None) -> Tensor:
        """"""

        assert self.input_has_edge_channel == (xe is not None)

        # propagate_type: (x: PairTensor)
        out = self.propagate(adj, x=x, xe=xe)

        # lin layer
        out = self.lin(out)
        if self.normalize:
            out = self.bn(out)
            
        return out

    def message_and_aggregate(self, adj: SparseTensor,
                              x: PairTensor, xe: OptTensor) -> Tensor:

        xu, xv = x
        adj = adj.set_value(None, layout=None)
        
        ## Node V to node U
        if self.flow == 'v->u':
            # messages node to node
            msg_v2u_list = [matmul(adj, xv, reduce=ag) for ag in self.agg]
            
            # messages edge to node
            if xe is not None:
                msg_e2u_list = [scatter(xe, adj.storage.row(), dim=0, reduce=ag) for ag in self.agg]
            
            # collect all msg
            if xe is not None:
                if self.node_self_loop:
                    if xu.shape[0] != msg_e2u_list[0].shape[0]:
                        print(f"xu: {xu.shape} | msg_v2u : {msg_v2u_list[0].shape} | msg_e2u_sum : {msg_e2u_list[0].shape}")
                    msg_2u = torch.cat((xu,
                                        *msg_v2u_list,
                                        *msg_e2u_list),
                                        dim=1)
                else:
                    msg_2u = torch.cat((*msg_v2u_list,
                                        *msg_e2u_list),
                                        dim=1)
            else:
                if self.node_self_loop:
                    msg_2u = torch.cat((xu, 
                                        *msg_v2u_list),
                                        dim=1)
                else:
                    msg_2u = torch.cat((*msg_v2u_list,),
                                        dim=1)

            return msg_2u
        
        ## Node U to node V
        else:
            msg_u2v_list = [matmul(adj.t(), xu, reduce=ag) for ag in self.agg]
            
            # messages edge to node
            if xe is not None:
                msg_e2v_list = [scatter(xe, adj.storage.col(), dim=0, reduce=ag) for ag in self.agg]

            # collect all msg (including self loop)
            if xe is not None:
                if self.node_self_loop:
                    msg_2v = torch.cat((xv, 
                                        *msg_u2v_list,
                                        *msg_e2v_list),
                                        dim=1)
                else:
                    msg_2v = torch.cat((*msg_u2v_list,
                                        *msg_e2v_list),
                                        dim=1)
            else:
                if self.node_self_loop:
                    msg_2v = torch.cat((xv, 
                                        *msg_u2v_list),
                                        dim=1)
                else:
                    msg_2v = torch.cat((*msg_u2v_list,),
                                        dim=1)

            return msg_2v


class BEANConvEdge(MessagePassing):
    
    def __init__(self, in_channels: Tuple[int, int, Optional[int]],
                 out_channels: int, 
                 node_self_loop: bool = True,
                 normalize: bool = True,
                 bias: bool = True, **kwargs):  

        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.node_self_loop = node_self_loop
        self.normalize = normalize

        self.input_has_edge_channel = (len(in_channels) == 3)
        
        if self.input_has_edge_channel:
            self.in_channels_e = in_channels[0] + in_channels[1] + in_channels[2]
        else:
            self.in_channels_e = in_channels[0] + in_channels[1]

        self.lin_e = Linear(self.in_channels_e, out_channels, bias=bias)

        if normalize:
            self.bn_e = nn.BatchNorm1d(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_e.reset_parameters()

    def forward(self, x: PairTensor, adj: SparseTensor, 
                xe: Tensor) -> Tensor:
        """"""

        # propagate_type: (x: PairTensor)
        out_e = self.propagate(adj, x=x, xe=xe)

        # lin layer
        out_e = self.lin_e(out_e)

        if self.normalize:
            out_e = self.bn_e(out_e)
        
        return out_e

    def message_and_aggregate(self, adj: SparseTensor,
                              x: PairTensor, xe: OptTensor) -> Tensor:

        xu, xv = x
        adj = adj.set_value(None, layout=None)
        
        # collect all msg (including self loop)
        if xe is not None:
            msg_2e = torch.cat((xe, 
                                xu[adj.storage.row()],
                                xv[adj.storage.col()]),
                                dim=1)         
        else:
            msg_2e = torch.cat((xu[adj.storage.row()],
                                    xv[adj.storage.col()]),
                                    dim=1)  

        return msg_2e