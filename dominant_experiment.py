# Copyright 2021 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import sys

from sklearn.metrics import roc_curve, precision_recall_curve, auc

from data_finefoods import load_graph

base_path = '/workspace/'
sys.path.insert(0, base_path) 

import argparse
import os

import torch
from torch_geometric.data import Data
from torch_scatter import scatter

from utils.seed import seed_all

# train a dominant detector
from pygod.models import DOMINANT

# %% args

parser = argparse.ArgumentParser(description='DOMINANT')
parser.add_argument("--name", type=str, default="wikipedia_anomaly",
                    help="name")
parser.add_argument("--key", type=str, default="graph_anomaly_list",
                    help="key to the data")
parser.add_argument("--id", type=int, default=0,
                    help="id to the data")
parser.add_argument("--n-epoch", type=int, default=200,
                    help="number of epoch")      
parser.add_argument("--num-neighbors", type=int, default=-1,
                    help="number of neighbors for node")   
parser.add_argument("--batch-size", type=int, default=0,
                    help="batch size")
parser.add_argument('--alpha', type=float, default=0.8, help='balance parameter')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--gpu', type=int, default=0, help='gpu number')


args1 = vars(parser.parse_args())

args2 = {
    "seed" : 0,
    "hidden_channels" : 32,
    "dropout_prob" : 0.0,
}

args = {**args1, **args2}

seed_all(args['seed'])

data_dir = '/workspace/results/'

# %% data
data = load_graph(args['name'], args['key'], args['id'])

u_ch = data.xu.shape[1]
v_ch = data.xv.shape[1]
e_ch = data.xe.shape[1]

print(f"Data dimension: U node = {data.xu.shape}; V node = {data.xv.shape}; E edge = {data.xe.shape}; \n")

# %% model

xu, xv = data.xu, data.xv
xe, adj = data.xe, data.adj
yu, yv, ye = data.yu, data.yv, data.ye


# %% to homogen
nu = xu.shape[0]
nv = xv.shape[0]
nn = nu + nv

# to homogen
row_h = torch.cat([adj.storage.row(), adj.storage.col() + nu])
col_h = torch.cat([adj.storage.col() + nu, adj.storage.row()])
edge_index_h = torch.stack([row_h, col_h])
xuh = torch.cat([scatter(xe, adj.storage.row(), dim=0, reduce='max'), scatter(xe, adj.storage.row(), dim=0, reduce='mean')], dim=1)
xvh = torch.cat([scatter(xe, adj.storage.col(), dim=0, reduce='max'), scatter(xe, adj.storage.col(), dim=0, reduce='mean')], dim=1)
xh = torch.cat([xuh, xvh], dim=0)
yh = torch.cat([yu, yv], dim=0)
data_h = Data(x=xh, edge_index=edge_index_h, y=yh)

# %% model

device = torch.device(f'cuda:{args["gpu"]}' if torch.cuda.is_available() else 'cpu')
model = DOMINANT(hid_dim=args['hidden_channels'], num_layers=4,
            dropout=args['dropout_prob'], alpha=args['alpha'], 
            epoch=args['n_epoch'], lr=args['lr'], verbose=True, gpu=args['gpu'],
            batch_size=args['batch_size'], num_neigh=args['num_neighbors'])

print(args)
print()

def auc_eval(pred, y):

    rc_curve = roc_curve(y, pred)
    pr_curve = precision_recall_curve(y, pred)
    roc_auc = auc(rc_curve[0], rc_curve[1])
    pr_auc = auc(pr_curve[1], pr_curve[0])

    return roc_auc, pr_auc, rc_curve, pr_curve

# %% run training

print('ready to run')

model.fit(data_h, yh)
score = model.decision_scores_

score_u = score[:nu]
score_v = score[nu:]
score_e_u = score_u[adj.storage.row().numpy()]
score_e_v = score_v[adj.storage.col().numpy()]
score_e = (score_e_u + score_e_v) / 2

u_roc_auc, u_pr_auc, u_rc_curve, u_pr_curve = auc_eval(score_u, yu.numpy())
v_roc_auc, v_pr_auc, v_rc_curve, v_pr_curve = auc_eval(score_v, yv.numpy())
e_roc_auc, e_pr_auc, e_rc_curve, e_pr_curve = auc_eval(score_e, ye.numpy())

print(f"Eval | " +
    f"u auc-roc: {u_roc_auc:.4f}, v auc-roc: {v_roc_auc:.4f}, e auc-roc: {e_roc_auc:.4f} | " + 
    f"u auc-pr {u_pr_auc:.4f}, v auc-pr {v_pr_auc:.4f}, e auc-pr {e_pr_auc:.4f}" 
    )

auc_metrics = {
    "u_roc_auc" : u_roc_auc,
    "u_pr_auc" : u_pr_auc,
    "v_roc_auc" : v_roc_auc,
    "v_pr_auc" : v_pr_auc,
    "e_roc_auc" : e_roc_auc,
    "e_pr_auc" : e_pr_auc,

    "u_roc_curve" : u_rc_curve,
    "u_pr_curve" : u_pr_curve,
    "v_roc_curve" : v_rc_curve,
    "v_pr_curve" : v_pr_curve,
    "e_roc_curve" : e_rc_curve,
    "e_pr_curve" : e_pr_curve,
}
anomaly_score ={
    "score_u" : score_u,
    "score_v" : score_v,
    "score_e" : score_e
}

model_stored = {
    "args" : args,
    "auc_metrics" : auc_metrics,
    "state_dict" : model.model.state_dict(),
}
output_stored = {
    "args" : args,
    "anomaly_score" : anomaly_score
}

print("Saving current results...")
torch.save(model_stored, os.path.join(data_dir, f"dominant-{args['name']}-{args['id']}-alpha-{args['alpha']}-model.th"))
torch.save(output_stored, os.path.join(data_dir, f"dominant-{args['name']}-{args['id']}-alpha-{args['alpha']}-output.th"))


print()
print(args)

