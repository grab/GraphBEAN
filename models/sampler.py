# Copyright 2021 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import torch
from torch import Tensor

import math

from torch_sparse import SparseTensor
from torch_sparse.storage import SparseStorage

from typing import List, NamedTuple, Optional, Tuple, Union

from utils.sparse_combine import spadd
from utils.sprand import sprand


class EdgePredictionSampler:
    def __init__(
        self,
        adj: SparseTensor,
        n_random: Optional[int] = None,
        mult: Optional[float] = 2.0,
    ):
        self.adj = adj

        if n_random is None:
            n_pos = adj.nnz()
            n_random = mult * n_pos

        self.adj = adj
        self.n_random = n_random

    def sample(self):
        rnd_samples = sprand(self.adj.sparse_sizes(), self.n_random)
        rnd_samples.fill_value_(-1)
        rnd_samples = rnd_samples.to(self.adj.device())

        pos_samples = self.adj.fill_value(2)

        samples = spadd(rnd_samples, pos_samples)
        samples.set_value_(
            torch.minimum(
                samples.storage.value(), torch.ones_like(samples.storage.value())
            ),
            layout="coo",
        )

        return samples


### REGION Neighbor Sampling
def sample_v_given_u(
    adj: SparseTensor,
    u_indices: Tensor,
    prev_v: Tensor,
    num_neighbors: int,
    replace=False,
) -> Tuple[SparseTensor, Tensor]:

    # to homogenous adjacency
    nu, nv = adj.sparse_sizes()
    adj_h = SparseTensor(
        row=adj.storage.row(),
        col=adj.storage.col() + nu,
        value=adj.storage.value(),
        sparse_sizes=(nu + nv, nu + nv),
    )

    res_adj_h, res_id = adj_h.sample_adj(
        torch.cat([u_indices, prev_v + nu]),
        num_neighbors=num_neighbors,
        replace=replace,
    )

    ni = len(u_indices)
    v_indices = res_id[ni:] - nu
    res_adj = res_adj_h[:ni, ni:]

    return res_adj, v_indices


def sample_u_given_v(
    adj: SparseTensor,
    v_indices: Tensor,
    prev_u: Tensor,
    num_neighbors: int,
    replace=False,
) -> Tuple[SparseTensor, Tensor]:

    # to homogenous adjacency
    res_adj_t, u_indices = sample_v_given_u(
        adj.t(), v_indices, prev_u, num_neighbors=num_neighbors, replace=replace
    )

    return res_adj_t.t(), u_indices


class DirectedAdj(NamedTuple):
    adj: SparseTensor
    u_id: Tensor
    v_id: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]
    flow: str

    def to(self, *args, **kwargs):
        adj = self.adj.to(*args, **kwargs)
        u_id = self.u_id.to(*args, **kwargs)
        v_id = self.v_id.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return DirectedAdj(adj, u_id, v_id, e_id, self.size, self.flow)


class BEANAdjacency(NamedTuple):
    adj_v2u: DirectedAdj
    adj_u2v: DirectedAdj
    adj_e: Optional[DirectedAdj]

    def to(self, *args, **kwargs):
        adj_v2u = self.adj_v2u.to(*args, **kwargs)
        adj_u2v = self.adj_u2v.to(*args, **kwargs)
        adj_e = None
        if self.adj_e is not None:
            adj_e = self.adj_e.to(*args, **kwargs)
        return BEANAdjacency(adj_v2u, adj_u2v, adj_e)


class BipartiteNeighborSampler(torch.utils.data.DataLoader):
    def __init__(
        self,
        adj: SparseTensor,
        n_layers: int,
        num_neighbors_u: Union[int, List[int]],
        num_neighbors_v: Union[int, List[int]],
        base: str = "u",
        n_other_node: int = -1,
        **kwargs
    ):

        adj = adj.to("cpu")

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        self.adj = adj
        self.n_layers = n_layers
        self.base = base
        self.n_other_node = n_other_node

        if isinstance(num_neighbors_u, int):
            num_neighbors_u = [num_neighbors_u for _ in range(n_layers)]
        if isinstance(num_neighbors_v, int):
            num_neighbors_v = [num_neighbors_v for _ in range(n_layers)]
        self.num_neighbors_u = num_neighbors_u
        self.num_neighbors_v = num_neighbors_v

        if base == "u":  # start from u
            item_idx = torch.arange(adj.sparse_size(0))
        elif base == "v":  # start from v instead
            item_idx = torch.arange(adj.sparse_size(1))
        elif base == "e":  # start from e instead
            item_idx = torch.arange(adj.nnz())
        else:  # start from u default
            item_idx = torch.arange(adj.sparse_size(0))

        value = torch.arange(adj.nnz())
        adj = adj.set_value(value, layout="coo")
        self.__val__ = adj.storage.value()

        # transpose of adjacency
        self.adj = adj
        self.adj_t = adj.t()

        # homogenous graph adjacency matrix
        self.nu, self.nv = self.adj.sparse_sizes()
        self.adj_homogen = SparseTensor(
            row=self.adj.storage.row(),
            col=self.adj.storage.col() + self.nu,
            value=self.adj.storage.value(),
            sparse_sizes=(self.nu + self.nv, self.nu + self.nv),
        )
        self.adj_t_homogen = SparseTensor(
            row=self.adj_t.storage.row(),
            col=self.adj_t.storage.col() + self.nv,
            value=self.adj_t.storage.value(),
            sparse_sizes=(self.nu + self.nv, self.nu + self.nv),
        )

        super(BipartiteNeighborSampler, self).__init__(
            item_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs
        )

    def sample_v_given_u(
        self, u_indices: Tensor, prev_v: Tensor, num_neighbors: int
    ) -> Tuple[SparseTensor, Tensor]:

        res_adj_h, res_id = self.adj_homogen.sample_adj(
            torch.cat([u_indices, prev_v + self.nu]),
            num_neighbors=num_neighbors,
            replace=False,
        )

        ni = len(u_indices)
        v_indices = res_id[ni:] - self.nu
        res_adj = res_adj_h[:ni, ni:]

        return res_adj, v_indices

    def sample_u_given_v(
        self, v_indices: Tensor, prev_u: Tensor, num_neighbors: int
    ) -> Tuple[SparseTensor, Tensor]:

        # start = time.time()
        res_adj_h, res_id = self.adj_t_homogen.sample_adj(
            torch.cat([v_indices, prev_u + self.nv]),
            num_neighbors=num_neighbors,
            replace=False,
        )
        # print(f"adjoint sampling : {time.time() - start} s")

        ni = len(v_indices)
        u_indices = res_id[ni:] - self.nv
        res_adj = res_adj_h[:ni, ni:]

        return res_adj.t(), u_indices

    def adjacency_from_samples(
        self, adj: SparseTensor, u_id: Tensor, v_id: Tensor, flow: str
    ) -> DirectedAdj:

        e_id = adj.storage.value()
        size = adj.sparse_sizes()
        if self.__val__ is not None:
            adj.set_value_(self.__val__[e_id], layout="coo")

        return DirectedAdj(adj, u_id, v_id, e_id, size, flow)

    def combine_adjacency(
        self, v2u_adj: SparseTensor, u2v_adj: SparseTensor, e_adj: SparseTensor
    ) -> SparseTensor:

        # start = time.time()
        nu = u2v_adj.sparse_size(0)
        nv = v2u_adj.sparse_size(1)

        row = torch.cat(
            [e_adj.storage.row(), v2u_adj.storage.row(), u2v_adj.storage.row()], dim=-1
        )
        col = torch.cat(
            [e_adj.storage.col(), v2u_adj.storage.col(), u2v_adj.storage.col()], dim=-1
        )
        value = torch.cat(
            [e_adj.storage.value(), v2u_adj.storage.value(), u2v_adj.storage.value()],
            dim=0,
        )
        fl = torch.cat(
            [
                torch.ones(e_adj.nnz()),
                2 * torch.ones(v2u_adj.nnz()),
                4 * torch.ones(u2v_adj.nnz()),
            ]
        )

        storage = SparseStorage(
            row=row, col=col, value=value, sparse_sizes=(nu, nv), is_sorted=False
        )
        storage = storage.coalesce(reduce="mean")

        fl_storage = SparseStorage(
            row=row, col=col, value=fl, sparse_sizes=(nu, nv), is_sorted=False
        )
        fl_storage = fl_storage.coalesce(reduce="sum")

        res = SparseTensor.from_storage(storage)
        flag = SparseTensor.from_storage(fl_storage)

        # print(f"combine adj : {time.time() - start} s")

        return res, flag

    def sample(self, batch):

        # start = time.time()

        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        # calculate batch_size for another node
        if self.n_other_node == -1 and self.base in ["u", "v"]:
            # do proportional
            nu, nv = self.adj.sparse_sizes()
            if self.base == "u":
                self.n_other_node = int(math.ceil((nv / nu) * batch_size))
            elif self.base == "v":
                self.n_other_node = int(math.ceil((nu / nv) * batch_size))

        ## get the other indices
        empty_list = torch.tensor([], dtype=torch.long)
        if self.base == "u":
            # get the base node for v
            u_indices = batch
            res_adj, res_id = self.sample_v_given_u(
                u_indices, empty_list, num_neighbors=self.num_neighbors_u[0]
            )
            rand_id = torch.randperm(len(res_id))[: self.n_other_node]
            v_indices = res_id[rand_id]
            e_adj = res_adj[:, rand_id]
        elif self.base == "v":
            # get the base node for u
            v_indices = batch
            res_adj, res_id = self.sample_u_given_v(
                v_indices, empty_list, num_neighbors=self.num_neighbors_v[0]
            )
            rand_id = torch.randperm(len(res_id))[: self.n_other_node]
            u_indices = res_id[rand_id]
            e_adj = res_adj[rand_id, :]
        elif self.base == "e":
            # get the base node for u and v
            row = self.adj.storage.row()[batch]
            col = self.adj.storage.col()[batch]
            unique_row, invidx_row = torch.unique(row, return_inverse=True)
            unique_col, invidx_col = torch.unique(col, return_inverse=True)

            reindex_row_id = torch.arange(len(unique_row))
            reindex_col_id = torch.arange(len(unique_col))
            reindex_row = reindex_row_id[invidx_row]
            reindex_col = reindex_col_id[invidx_col]

            e_adj = SparseTensor(row=reindex_row, col=reindex_col, value=batch)
            e_indices = batch
            u_indices = unique_row
            v_indices = unique_col

        # init results
        adjacencies = []
        e_flags = []

        ## for subsequent layers
        for i in range(self.n_layers):

            # v -> u
            u_adj, next_v_indices = self.sample_v_given_u(
                u_indices, prev_v=v_indices, num_neighbors=self.num_neighbors_u[i]
            )
            dir_adj_v2u = self.adjacency_from_samples(
                u_adj, u_indices, next_v_indices, "v->u"
            )

            # u -> v
            v_adj, next_u_indices = self.sample_u_given_v(
                v_indices, prev_u=u_indices, num_neighbors=self.num_neighbors_v[i]
            )
            dir_adj_u2v = self.adjacency_from_samples(
                v_adj, next_u_indices, v_indices, "u->v"
            )

            # u -> e <- v
            dir_adj_e = self.adjacency_from_samples(
                e_adj, u_indices, v_indices, "u->e<-v"
            )

            # add them to the list
            adjacencies.append(BEANAdjacency(dir_adj_v2u, dir_adj_u2v, dir_adj_e))

            # for next iter
            e_adj, e_flag = self.combine_adjacency(
                v2u_adj=u_adj, u2v_adj=v_adj, e_adj=e_adj
            )
            u_indices = next_u_indices
            v_indices = next_v_indices
            e_flags.append(e_flag)

        # flip the order
        adjacencies = adjacencies[0] if len(adjacencies) == 1 else adjacencies[::-1]
        e_flags = e_flags[0] if len(e_flags) == 1 else e_flags[::-1]

        # get e_indices
        e_indices = e_adj.storage.value()

        # print(f"sampling : {time.time() - start} s")

        return batch_size, (u_indices, v_indices, e_indices), adjacencies, e_flags


class EdgeLoader(torch.utils.data.DataLoader):
    def __init__(self, adj: SparseTensor, **kwargs):

        edge_idx = torch.arange(adj.nnz())
        self.adj = adj

        super().__init__(edge_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):

        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        row = self.adj.storage.row()[batch]
        col = self.adj.storage.col()[batch]
        if self.adj.storage.has_value():
            val = self.adj.storage.col()[batch]
        else:
            val = batch

        # get unique row, col & idx
        unique_row, invidx_row = torch.unique(row, return_inverse=True)
        unique_col, invidx_col = torch.unique(col, return_inverse=True)

        reindex_row_id = torch.arange(len(unique_row))
        reindex_col_id = torch.arange(len(unique_col))

        reindex_row = reindex_row_id[invidx_row]
        reindex_col = reindex_col_id[invidx_col]

        adj = SparseTensor(row=reindex_row, col=reindex_col, value=val)
        e_id = batch
        u_id = unique_row
        v_id = unique_col

        adj_e = DirectedAdj(adj, u_id, v_id, e_id, adj.sparse_sizes(), "u->e<-v")

        return adj_e
