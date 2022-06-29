# Copyright 2021 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import torch
from torch import Tensor

from torch_sparse import SparseTensor

from typing import Optional, Tuple

from torch_sparse import SparseTensor
from torch_sparse.storage import SparseStorage


def spadd(A: SparseTensor, B: SparseTensor, op: str = "add") -> SparseTensor:
    assert A.sparse_sizes() == B.sparse_sizes()

    m, n = A.sparse_sizes()

    row = torch.cat([A.storage.row(), B.storage.row()], dim=-1)
    col = torch.cat([A.storage.col(), B.storage.col()], dim=-1)
    value = torch.cat([A.storage.value(), B.storage.value()], dim=0)

    storage = SparseStorage(
        row=row, col=col, value=value, sparse_sizes=(m, n), is_sorted=False
    )
    storage = storage.coalesce(reduce=op)

    return SparseTensor.from_storage(storage)


## sparse combine
def sparse_combine(
    a: SparseTensor, b: SparseTensor, flag_mult: Optional[Tuple[int, int]] = (1, 2)
) -> Tuple[SparseTensor, SparseTensor]:

    res = spadd(a, b, op="mean")

    # flag where the source come from
    flag = spadd(a.fill_value(flag_mult[0]), b.fill_value(flag_mult[1]))

    return res, flag


## sparse combine
def sparse_combine3(
    a: SparseTensor, b: SparseTensor, c: SparseTensor
) -> Tuple[SparseTensor, SparseTensor, SparseTensor]:

    flag_mult = (1, 2, 4)
    res = spadd(spadd(a, b, op="mean"), c, op="mean")

    # flag where the source come from
    flag = spadd(
        spadd(a.fill_value(flag_mult[0]), b.fill_value(flag_mult[1])),
        c.fill_value(flag_mult[2]),
    )

    return res, flag


def sparse_combine3a(
    a: SparseTensor, b: SparseTensor, c: SparseTensor
) -> Tuple[SparseTensor, SparseTensor, SparseTensor]:

    flag_mult = (1, 2, 4)

    # add values
    d = SparseTensor.from_torch_sparse_coo_tensor(
        a.to_torch_sparse_coo_tensor()
        + b.to_torch_sparse_coo_tensor()
        + c.to_torch_sparse_coo_tensor()
    )
    # add non zeros
    e = SparseTensor.from_torch_sparse_coo_tensor(
        a.fill_value(1).to_torch_sparse_coo_tensor()
        + b.fill_value(1).to_torch_sparse_coo_tensor()
        + c.fill_value(1).to_torch_sparse_coo_tensor()
    )

    # rmove duplicate values
    val = (d.storage.value() / e.storage.value()).long()
    res = d.set_value(val, layout="coo")

    # flag where the source come from
    flag = SparseTensor.from_torch_sparse_coo_tensor(
        a.fill_value(flag_mult[0]).to_torch_sparse_coo_tensor()
        + b.fill_value(flag_mult[1]).to_torch_sparse_coo_tensor()
        + c.fill_value(flag_mult[2]).to_torch_sparse_coo_tensor()
    )

    return res, flag


def xe_split3(
    xe: Tensor,
    flag: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:

    # flag_mult = (1,2,4)

    a_idx = (flag == 1) | (flag == 3) | (flag == 5) | (flag == 7)
    b_idx = (flag == 2) | (flag == 3) | (flag == 6) | (flag == 7)
    c_idx = (flag == 4) | (flag == 5) | (flag == 6) | (flag == 7)

    xe_a = xe[a_idx]
    xe_b = xe[b_idx]
    xe_c = xe[c_idx]

    return xe_a, xe_b, xe_c
