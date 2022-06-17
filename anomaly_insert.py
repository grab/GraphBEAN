# Copyright 2021 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import torch

import numpy as np
from scipy.stats import truncnorm
from torch_sparse import SparseTensor

from models.data import BipartiteData

from typing import Tuple, Union

# %% features outliers

# features outside confidence interval
def outside_cofidence_interval(x: torch.Tensor, prop_sample=0.1, prop_feat=0.3, std_cutoff=3.0, mu=None, sigm=None):
    n, m = x.shape
    ns = int(np.ceil(prop_sample * n))
    ms = int(np.ceil(prop_feat * m))
    
    # random outlier from truncated normal
    left_side = truncnorm.rvs(-np.inf, -std_cutoff, size=ns*ms)
    right_side = truncnorm.rvs(std_cutoff, np.inf, size=ns*ms)
    lr_flag = np.random.randint(2, size=ns*ms)
    random_outliers = lr_flag * left_side + (1 - lr_flag) * right_side

    # determine which sample & features that are randomized
    feat_idx = np.random.rand(ns, m).argsort(axis=1)[:, :ms]
    sample_idx = np.random.choice(n, ns, replace=False)
    row_idx = np.tile(sample_idx[:, None], (1, ms)).flatten()
    col_idx = feat_idx.flatten()

    # calculate mean and variance
    xr = x.cpu().numpy()
    if mu is None:
        mu = xr.mean(axis=0)
    if sigm is None:
        sigm = xr.std(axis=0)

    # replace the value with outliers
    random_outliers = random_outliers * sigm[col_idx] + mu[col_idx]
    xr[(row_idx, col_idx)] = random_outliers

    # anomaly
    anomaly_label = torch.zeros(n).long()
    anomaly_label[sample_idx] = 1 

    return torch.Tensor(xr), anomaly_label, row_idx, col_idx


# add scaled gaussian noise
def scaled_gaussian_noise(x: torch.Tensor, scale=3.0, min_dist_rel=3.0, filter=True, mu=None, sigm=None):
    
    # calculate mean and variance
    if mu is None:
        mu = x.mean(dim=0)
    if sigm is None:
        sigm = x.std(dim=0)

    # noise
    noise = torch.randn(x.shape) * sigm * scale
    outlier = x + noise
    closest_dist = torch.cdist(outlier, x, p=1).min(dim=1)[0]
    if filter:
        anomaly_label = (closest_dist/x.shape[1] > min_dist_rel).long()
        # replace the value with outliers
        xr = anomaly_label[:, None] * outlier  +  (1 - anomaly_label[:, None]) * x 
    else:
        anomaly_label = torch.ones(x.shape[0]).long()
        xr = outlier

    return xr, anomaly_label


# %% structure outliers
def dense_block(adj: SparseTensor, xe: torch.Tensor, ye=None,
        num_nodes: Union[int, Tuple[int, int]] = 5, num_group: int = 2, connected_prop = 1.0, 
        feature_anomaly = False, feature_anomaly_type="outside_ci", **kwargs):

    n, m = adj.sparse_sizes()
    ne = xe.shape[0]

    if isinstance(num_nodes, int):
        num_nodes = (num_nodes, num_nodes)

    row = adj.storage.row()
    col = adj.storage.col()
    ids = torch.stack([row, col])

    outlier_row = torch.zeros(0).long()
    outlier_col = torch.zeros(0).long()

    for i in range(num_group):
        rid = np.random.choice(n, num_nodes[0], replace=False)
        cid = np.random.choice(m, num_nodes[1], replace=False)

        # all nodes are connected
        rows_id = torch.tensor(np.tile(rid[:,None], (1,num_nodes[1])).flatten())
        cols_id = torch.tensor(np.tile(cid, num_nodes[0]))

        # partially dense connection
        if connected_prop < 1.0:
            n_connected = rows_id.shape[0]
            n_taken = int(np.ceil(connected_prop * n_connected))
            taken_id = np.random.choice(n_connected, n_taken, replace=False)

            rows_id = rows_id[taken_id]
            cols_id = cols_id[taken_id]

        # add to the graph
        outlier_row = torch.cat([outlier_row, rows_id])
        outlier_col = torch.cat([outlier_col, cols_id])

    # only unique ids
    outlier_ids = torch.stack([outlier_row, outlier_col]).unique(dim=1)

    # find additional ids that is not in the current adj
    ids_all, inv, count = torch.cat([ids, outlier_ids], dim=1).unique(dim=1, return_counts=True, return_inverse=True)
    ids_duplicate = ids_all[:, count > 1]
    ids_2, count_2 = torch.cat([outlier_ids, ids_duplicate], dim=1).unique(dim=1, return_counts=True)
    ids_additional = ids_2[:, count_2 == 1]

    # anomalous label for the original
    label_orig = (count[inv][:ne] > 1).long()

    ## features
    n_add = ids_additional.shape[1]
    # random features for the new edges
    add_ids = np.random.choice(ne, n_add, replace=False)
    xe_add = xe[add_ids, :]

    # inject feature anomaly
    xe2 = xe.clone()
    if feature_anomaly:
        mu = xe.mean(dim=0).numpy()
        sigm = xe.std(dim=0).numpy()
        kwargs["mu"] = mu
        kwargs["sigm"] = sigm    

        if feature_anomaly_type == "outside_ci":
            kwargs["prop_sample"] = 1.0
            xe_add = outside_cofidence_interval(xe_add, **kwargs)[0]
            if label_orig.sum() > 0:
                xe2[label_orig == 1, :] = outside_cofidence_interval(xe[label_orig == 1, :], **kwargs)[0]
            else:
                xe2 = xe
        elif feature_anomaly_type == "scaled_gaussian":
            kwargs["filter"] = False
            xe_add = scaled_gaussian_noise(xe_add, **kwargs)[0]
            if label_orig.sum() > 0:
                xe2[label_orig == 1, :] = scaled_gaussian_noise(xe[label_orig == 1, :], **kwargs)[0]
            else:
                xe2 = xe

    # combine with the previous label if given
    ye2 = label_orig if ye is None else torch.logical_or(ye, label_orig).long() 

    # attach xe and label to value
    ids_cmb = torch.cat([ids, ids_additional], dim=1)
    xe_cmb = torch.cat([xe2, xe_add], dim=0)
    ye_cmb = torch.cat([ye2, torch.ones(n_add).long()])
    label_cmb = torch.cat([label_orig, torch.ones(n_add).long()])
    value_cmb = torch.cat([xe_cmb, ye_cmb[:,None], label_cmb[:,None]], dim=1)
    
    # get result
    adj_new = SparseTensor(row=ids_cmb[0], col=ids_cmb[1], value=value_cmb).coalesce()
    value_new = adj_new.storage.value()
    xe_new = value_new[:,:-2]
    ye_new = value_new[:,-2].long()
    label = value_new[:,-1].long()
    adj_new.storage._value = None

    return adj_new, xe_new, ye_new, label


# %% graph, insert anomaly

def inject_feature_anomaly(data: BipartiteData, node_anomaly=True, edge_anomaly=True, 
                           feature_anomaly_type="outside_ci", **kwargs):
    
    if node_anomaly:
        if feature_anomaly_type == "outside_ci":
            xu, yu2, _, _ = outside_cofidence_interval(data.xu, **kwargs)
            xv, yv2, _, _ = outside_cofidence_interval(data.xv, **kwargs) 
        elif feature_anomaly_type == "scaled_gaussian":
            xu, yu2 = scaled_gaussian_noise(data.xu, **kwargs)
            xv, yv2 = scaled_gaussian_noise(data.xv, **kwargs)
        yu = torch.logical_or(data.yu, yu2).long() if hasattr(data, 'yu') else yu2
        yv = torch.logical_or(data.yv, yv2).long() if hasattr(data, 'yv') else yv2

    else:
        xu = data.xu
        xv = data.xv
        yu = data.yu if hasattr(data, 'yu') else None
        yv = data.yv if hasattr(data, 'yv') else None

    if edge_anomaly:
        if feature_anomaly_type == "outside_ci":
            xe, ye2, _, _ = outside_cofidence_interval(data.xe, **kwargs)
        elif feature_anomaly_type == "scaled_gaussian":
            xe, ye2 = scaled_gaussian_noise(data.xe, **kwargs)
        ye = torch.logical_or(data.ye, ye2).long() if hasattr(data, 'ye') else ye2
    else:
        xe = data.xe
        ye = data.ye if hasattr(data, 'ye') else None

    data_new = BipartiteData(data.adj, xu=xu, xv=xv, xe=xe, yu=yu, yv=yv, ye=ye)    

    return data_new


def inject_dense_block_anomaly(data: BipartiteData, **kwargs):
    kwargs["feature_anomaly"] = False
    ye = data.ye if hasattr(data, 'ye') else None
    adj_new, xe_new, ye_new, label = dense_block(data.adj, data.xe, ye=ye, **kwargs)

    yu = torch.zeros(data.xu.shape[0]).long()
    yu[adj_new.storage.row()[label == 1].unique()] = 1

    yv = torch.zeros(data.xv.shape[0]).long()
    yv[adj_new.storage.col()[label == 1].unique()] = 1

    data_new = BipartiteData(adj_new, xu=data.xu, xv=data.xv, xe=xe_new)
    data_new.ye = ye_new
    data_new.yu = torch.logical_or(data.yu, yu).long() if hasattr(data, 'yu') else yu
    data_new.yv = torch.logical_or(data.yv, yv).long() if hasattr(data, 'yv') else yv
    
    return data_new

def inject_dense_block_and_feature_anomaly(data: BipartiteData, 
        node_feature_anomaly=False, edge_feature_anomaly=True, **kwargs):

    kwargs["feature_anomaly"] = edge_feature_anomaly
    if "feature_anomaly_type" not in kwargs:
        kwargs["feature_anomaly_type"] = "outside_ci"

    ye = data.ye if hasattr(data, 'ye') else None
    adj_new, xe_new, ye_new, label = dense_block(data.adj, data.xe, ye=ye, **kwargs)

    yu = torch.zeros(data.xu.shape[0]).long()
    yu[adj_new.storage.row()[label == 1].unique()] = 1

    yv = torch.zeros(data.xv.shape[0]).long()
    yv[adj_new.storage.col()[label == 1].unique()] = 1

    # also node feature anomaly
    if node_feature_anomaly:
        
        # args
        kw2 = {}

        # xu
        xu = data.xu
        mu = xu.mean(dim=0).numpy()
        sigm = xu.std(dim=0).numpy()
        kw2["mu"] = mu
        kw2["sigm"] = sigm

        if kwargs["feature_anomaly_type"] == "outside_ci":
            kw2["prop_sample"] = 1.0
            if 'prop_feat' in kwargs: kw2['prop_feat'] = kwargs['prop_feat']
            if 'std_cutoff' in kwargs: kw2['std_cutoff'] = kwargs['std_cutoff']
            xu_new = xu.clone()
            xu_new[yu == 1, :] = outside_cofidence_interval(xu[yu == 1, :], **kw2)[0]
        elif kwargs["feature_anomaly_type"] == "scaled_gaussian":
            kw2["filter"] = False
            if 'scale' in kwargs: kw2['scale'] = kwargs['scale']
            if 'min_dist_rel' in kwargs: kw2['min_dist_rel'] = kwargs['min_dist_rel']
            xu_new = xu.clone()
            xu_new[yu == 1, :] = scaled_gaussian_noise(xu[yu == 1, :], **kw2)[0]
            
        # xv
        xv = data.xv
        mu = xv.mean(dim=0).numpy()
        sigm = xv.std(dim=0).numpy()
        kw2["mu"] = mu
        kw2["sigm"] = sigm

        if kwargs["feature_anomaly_type"] == "outside_ci":
            kw2["prop_sample"] = 1.0
            if 'prop_feat' in kwargs: kw2['prop_feat'] = kwargs['prop_feat']
            if 'std_cutoff' in kwargs: kw2['std_cutoff'] = kwargs['std_cutoff']
            xv_new = xv.clone()
            xv_new[yv == 1, :] = outside_cofidence_interval(xv[yv == 1, :], **kw2)[0]
        elif kwargs["feature_anomaly_type"] == "scaled_gaussian":
            kw2["filter"] = False
            if 'scale' in kwargs: kw2['scale'] = kwargs['scale']
            if 'min_dist_rel' in kwargs: kw2['min_dist_rel'] = kwargs['min_dist_rel']
            xv_new = xv.clone()
            xv_new[yv == 1, :] = scaled_gaussian_noise(xv[yv == 1, :], **kw2)[0]

        # data
        data_new = BipartiteData(adj_new, xu=xu_new, xv=xv_new, xe=xe_new)
        data_new.ye = ye_new
        data_new.yu = torch.logical_or(data.yu, yu).long() if hasattr(data, 'yu') else yu
        data_new.yv = torch.logical_or(data.yv, yv).long() if hasattr(data, 'yv') else yv

    else:
        data_new = BipartiteData(adj_new, xu=data.xu, xv=data.xv, xe=xe_new)
        data_new.ye = ye_new
        data_new.yu = torch.logical_or(data.yu, yu).long() if hasattr(data, 'yu') else yu
        data_new.yv = torch.logical_or(data.yv, yv).long() if hasattr(data, 'yv') else yv
    
    return data_new

# %% random anomaly

def choose(r, choices, thresholds):
    i = 0
    cm = thresholds[i]
    while i < len(choices):
        if r <= cm + 1e-9:
            selected = i
            break
        else:
            i += 1
            if i < len(choices):
                cm += thresholds[i]
            else:
                selected = len(choices) - 1
                break
    
    return choices[selected]


def inject_random_block_anomaly(data: BipartiteData, 
        num_group=40, num_nodes_range=(1, 12), num_nodes_range2=None, **kwargs):

    block_anomalies = ['full_dense_block', 'partial_full_dense_block'] #, 'none']
    feature_anomalies = ['outside_ci', 'scaled_gaussian', 'none']
    node_edge_feat_anomalies = ['node_only', 'edge_only', 'node_edge']

    block_anomalies_weight = [0.2, 0.8] #, 0.1]
    feature_anomalies_weight = [0.5, 0.4, 0.1]
    node_edge_feat_anomalies_weight = [0.1, 0.3, 0.6]

    data_new = BipartiteData(data.adj, xu=data.xu, xv=data.xv, xe=data.xe)

    # random anomaly
    for itg in range(num_group):

        print(f'it {itg}: ', end='')

        rnd = torch.rand(3)
        block_an = choose(rnd[0], block_anomalies, block_anomalies_weight)
        feature_an = choose(rnd[1], feature_anomalies, feature_anomalies_weight)
        node_edge_an = choose(rnd[2], node_edge_feat_anomalies, node_edge_feat_anomalies_weight)
        lr, rr, mr = num_nodes_range[0], num_nodes_range[1], num_nodes_range[0]+num_nodes_range[1]/2
        if num_nodes_range2 is not None:
            nn1 = int(np.minimum(np.maximum(lr, (torch.randn(1).item() * np.sqrt(mr)) + mr), rr + 1))
            lr2, rr2, mr2 = num_nodes_range2[0], num_nodes_range2[1], num_nodes_range2[0]+num_nodes_range2[1]/2
            nn2 = int(np.minimum(np.maximum(lr2, (torch.randn(1).item() * np.sqrt(mr2)) + mr2), rr2 + 1))
            num_nodes = (nn1, nn2)
        else:
            num_nodes = int(np.minimum(np.maximum(lr, (torch.randn(1).item() * np.sqrt(mr)) + mr), rr + 1))

        ## setup kwargs
        connected_prop = 1.0
        if block_an == 'partial_full_dense_block':
            connected_prop = np.minimum(np.maximum(0.2, (torch.randn(1).item() / 4) + 0.5), 1.0)

        prop_feat = np.minimum(np.maximum(0.1, (torch.randn(1).item() / 8) + 0.3), 0.9)
        std_cutoff = np.maximum(2.0, torch.randn(1).item() + 3.0)
        scale = np.maximum(2.0, torch.randn(1).item() + 3.0)

        ## inject anomaly
        node_feature_anomaly = None
        if block_an != 'none' and feature_an != 'none':
            node_feature_anomaly = False if node_edge_an == 'edge_only' else True
            edge_feature_anomaly = False if node_edge_an == 'node_only' else True

            if feature_an == 'outside_ci':
                data_new = inject_dense_block_and_feature_anomaly(data_new, node_feature_anomaly, edge_feature_anomaly,
                            num_group=1, num_nodes=num_nodes, connected_prop=connected_prop, 
                            feature_anomaly_type='outside_ci', prop_feat=prop_feat, std_cutoff=std_cutoff)

            elif feature_an == 'scaled_gaussian':
                data_new = inject_dense_block_and_feature_anomaly(data_new, node_feature_anomaly, edge_feature_anomaly,
                            num_group=1, num_nodes=num_nodes, connected_prop=connected_prop, 
                            feature_anomaly_type='scaled_gaussian', scale=scale)
            
        elif block_an != 'none' and feature_an == 'none':
            data_new = inject_dense_block_anomaly(data_new, 
                        num_group=1, num_nodes=num_nodes, connected_prop=connected_prop)

        elif block_an == 'none' and feature_an != 'none':
            node_anomaly = False if node_edge_an == 'edge_only' else True
            edge_anomaly = False if node_edge_an == 'node_only' else True

            if feature_an == 'outside_ci':
                data_new = inject_feature_anomaly(data_new, node_anomaly, edge_anomaly,
                            feature_anomaly_type='outside_ci', prop_feat=prop_feat, std_cutoff=std_cutoff)

            elif feature_an == 'scaled_gaussian':
                data_new = inject_feature_anomaly(data_new, node_anomaly, edge_anomaly,
                            feature_anomaly_type='scaled_gaussian', scale=scale)

        print(f'affected: yu = {data_new.yu.sum()}, yv = {data_new.yv.sum()}, ye = {data_new.ye.sum()}  ', end='')
        print(f'[{block_an}:{connected_prop:.2f},{feature_an},{num_nodes},{node_feature_anomaly}]')

    return data_new


