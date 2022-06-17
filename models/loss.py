# Copyright 2021 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

from typing import Dict, Tuple
import torch.nn.functional as F
from torch import Tensor

from torch_sparse import SparseTensor

def reconstruction_loss(xu: Tensor, xv: Tensor,
                        xe: Tensor, adj: SparseTensor,
                        edge_pred_samples: SparseTensor,
                        out: Dict[str, Tensor], 
                        xe_loss_weight: float = 1.0,
                        structure_loss_weight: float = 1.0) -> Tuple[Tensor, Dict[str, Tensor]]:
    # feature mse
    xu_loss = F.mse_loss(xu, out["xu"])
    xv_loss = F.mse_loss(xv, out["xv"])
    xe_loss = F.mse_loss(xe, out["xe"])
    feature_loss = xu_loss + xv_loss + xe_loss_weight * xe_loss
    
    # structure loss
    edge_gt = (edge_pred_samples.storage.value() > 0).float()
    structure_loss = F.binary_cross_entropy(out['eprob'],  edge_gt)

    loss = feature_loss + structure_loss_weight * structure_loss

    loss_component = {
        'xu': xu_loss,
        'xv': xv_loss,
        'xe': xe_loss,
        'e': structure_loss,
        'total': loss
    }

    return loss, loss_component
    
    
