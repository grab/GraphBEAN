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
from utils.seed import seed_all

from sklearn.ensemble import IsolationForest

# %% args

parser = argparse.ArgumentParser(description='IsolationForest')
parser.add_argument("--name", type=str, default="wikipedia_anomaly",
                    help="name")
parser.add_argument("--key", type=str, default="graph_anomaly_list",
                    help="key to the data")
parser.add_argument("--id", type=int, default=0,
                    help="id to the data")
   
args1 = vars(parser.parse_args())

args2 = {
    "seed" : 0,
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


def train_eval(x, y):
    clf = IsolationForest()
    clf.fit(x)
    score = -clf.score_samples(x)

    rc_curve = roc_curve(y, score)
    pr_curve = precision_recall_curve(y, score)
    roc_auc = auc(rc_curve[0], rc_curve[1])
    pr_auc = auc(pr_curve[1], pr_curve[0])

    return roc_auc, pr_auc, rc_curve, pr_curve



# %% isolation forest

u_roc_auc, u_pr_auc, u_rc_curve, u_pr_curve = train_eval(xu.numpy(), yu.numpy())
v_roc_auc, v_pr_auc, v_rc_curve, v_pr_curve = train_eval(xv.numpy(), yv.numpy())
e_roc_auc, e_pr_auc, e_rc_curve, e_pr_curve = train_eval(xe.numpy(), ye.numpy())

print(args)

print(f"Eval, " +
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

output_stored = {
    "args" : args,
    "auc_metrics" : auc_metrics,
}

print("Saving current results...")
torch.save(output_stored, os.path.join(data_dir, f"isoforest-{args['name']}-{args['id']}-output.th"))