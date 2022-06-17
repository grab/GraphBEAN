# Copyright 2021 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import sys

from data_finefoods import load_graph
from models.score import compute_evaluation_metrics

base_path = '/workspace/'
sys.path.insert(0, base_path) 

import time
from tqdm import tqdm
import argparse
import os

from torch.utils.tensorboard import SummaryWriter
import datetime

import torch

from models.data import BipartiteData
from models.net import GraphBEAN
from models.sampler import EdgePredictionSampler
from models.loss import reconstruction_loss
from models.score import compute_anomaly_score, edge_prediction_metric

from utils.seed import seed_all

# %% args

parser = argparse.ArgumentParser(description='GraphBEAN')
parser.add_argument("--name", type=str, default="wikipedia_anomaly",
                    help="name")
parser.add_argument("--key", type=str, default="graph_anomaly_list",
                    help="key to the data")
parser.add_argument("--id", type=int, default=0,
                    help="id to the data")
parser.add_argument("--n-epoch", type=int, default=200,
                    help="number of epoch")    
parser.add_argument("--scheduler-milestones", nargs='+', type=int, default=[],
                    help="scheduler milestone")        
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--score-agg', type=str, default='max', help='aggregation for node anomaly score')
parser.add_argument('--eta', type=float, default=0.2, help='structure loss weight')

args1 = vars(parser.parse_args())

args2 = {
    "hidden_channels" : 32,
    "latent_channels_u" : 32,
    "latent_channels_v" : 32,
    "edge_pred_latent" : 32,
    "n_layers_encoder" : 2,
    "n_layers_decoder" : 2,
    "n_layers_mlp" : 2,
    "dropout_prob" : 0.0,
    "gamma" : 0.2,
    "xe_loss_weight" : 1.0,
    "structure_loss_weight" : args1['eta'],
    "structure_loss_weight_anomaly_score" : args1['eta'],
    "iter_check" : 10,
    "seed" : 0,
    "neg_sampler_mult" : 5,
    'k_check' : 15,
    'tensorboard' : False,
    'progress_bar' : True
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphBEAN(
    in_channels=(u_ch, v_ch, e_ch),
    hidden_channels = args["hidden_channels"],
    latent_channels = (args["latent_channels_u"], args["latent_channels_v"]),
    edge_pred_latent = args["edge_pred_latent"],
    n_layers_encoder = args["n_layers_encoder"],
    n_layers_decoder = args["n_layers_decoder"],
    n_layers_mlp = args["n_layers_mlp"],
    dropout_prob = args["dropout_prob"]
    )

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args["scheduler_milestones"], gamma=args["gamma"])

xu, xv = data.xu.to(device), data.xv.to(device)
xe, adj = data.xe.to(device), data.adj.to(device)
yu, yv, ye = data.yu.to(device), data.yv.to(device), data.ye.to(device)


# sampler
sampler = EdgePredictionSampler(adj, mult=args["neg_sampler_mult"])

print(args)
print()

# %% train
def train(epoch):

    model.train()

    edge_pred_samples = sampler.sample()

    optimizer.zero_grad()
    out = model(xu, xv, xe, adj, edge_pred_samples)

    loss, loss_component = reconstruction_loss(xu, xv, xe, adj, edge_pred_samples, out, 
                    xe_loss_weight=args["xe_loss_weight"], structure_loss_weight=args["structure_loss_weight"])

    loss.backward()
    optimizer.step()
    scheduler.step()
    
    epred_metric = edge_prediction_metric(edge_pred_samples, out['eprob'])

    return loss, loss_component, epred_metric

# %% evaluate and store
def eval(epoch):

    # model.eval()

    start = time.time()

    # negative sampling
    edge_pred_samples = sampler.sample()

    with torch.no_grad():
        
        out = model(xu, xv, xe, adj, edge_pred_samples)
        
        loss, loss_component = reconstruction_loss(xu, xv, xe, adj, edge_pred_samples, out, 
                    xe_loss_weight=args["xe_loss_weight"], structure_loss_weight=args["structure_loss_weight"])

        epred_metric = edge_prediction_metric(edge_pred_samples, out['eprob'])

        anomaly_score = compute_anomaly_score(xu, xv, xe, adj, edge_pred_samples, out,
                        xe_loss_weight = args["xe_loss_weight"],
                        structure_loss_weight = args["structure_loss_weight_anomaly_score"])

        eval_metrics = compute_evaluation_metrics(anomaly_score, yu, yv, ye, agg=args["score_agg"])

    elapsed = time.time() - start


    print(f"Eval, loss: {loss:.4f}, " +
        f"u auc-roc: {eval_metrics['u_roc_auc']:.4f}, v auc-roc: {eval_metrics['v_roc_auc']:.4f}, e auc-roc: {eval_metrics['e_roc_auc']:.4f}, " + 
        f"u auc-pr {eval_metrics['u_pr_auc']:.4f}, v auc-pr {eval_metrics['v_pr_auc']:.4f}, e auc-pr {eval_metrics['e_pr_auc']:.4f} " + 
        f"> {elapsed:.2f}s"
        )

    if args['tensorboard']:
        tb.add_scalar('loss', loss, epoch)
        tb.add_scalar('u_roc_auc', eval_metrics['u_roc_auc'], epoch)
        tb.add_scalar('u_pr_auc', eval_metrics['u_pr_auc'], epoch)
        tb.add_scalar('v_roc_auc', eval_metrics['v_roc_auc'], epoch)
        tb.add_scalar('v_pr_auc', eval_metrics['v_pr_auc'], epoch)
        tb.add_scalar('e_roc_auc', eval_metrics['e_roc_auc'], epoch)
        tb.add_scalar('e_pr_auc', eval_metrics['e_pr_auc'], epoch)
       
    model_stored = {
        "args" : args,
        "loss" : loss,
        "loss_component" : loss_component,
        "epred_metric" : epred_metric,
        "eval_metrics" : eval_metrics,
        "loss_hist" : loss_hist,
        "loss_component_hist" : loss_component_hist,
        "epred_metric_hist" : epred_metric_hist,
        "state_dict" : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    output_stored = {
        "args" : args,
        "out" : out,
        "anomaly_score" : anomaly_score
    }

    print("Saving current results...")
    torch.save(model_stored, os.path.join(data_dir, f"graphbean-{args['name']}-{args['id']}-eta-{args['eta']}-structure-model.th"))
    torch.save(output_stored, os.path.join(data_dir, f"graphbean-{args['name']}-{args['id']}-eta-{args['eta']}-structure-output.th"))

    return loss, loss_component, epred_metric


# %% run training
loss_hist = []
loss_component_hist = []
epred_metric_hist = []

# tensor board
if args['tensorboard']:
    log_dir = "/logs/tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + args['name']
    tb = SummaryWriter(log_dir=log_dir, comment=args['name'])
check_counter = 0

eval(0)

for epoch in range(args["n_epoch"]):

    start = time.time()
    loss, loss_component, epred_metric = train(epoch)
    elapsed = time.time() - start

    loss_hist.append(loss)
    loss_component_hist.append(loss_component)
    epred_metric_hist.append(epred_metric)

    print(f"#{epoch:3d}, " +
          f"Loss: {loss:.4f} => xu: {loss_component['xu']:.4f}, xv: {loss_component['xv']:.4f}, " + 
          f"xe: {loss_component['xe']:.4f}, " + 
          f"e: {loss_component['e']:.4f} -> " + 
          f"[acc: {epred_metric['acc']:.3f}, f1: {epred_metric['f1']:.3f} -> " +
          f"prec: {epred_metric['prec']:.3f}, rec: {epred_metric['rec']:.3f}] " +
          f"> {elapsed:.2f}s"
          )
    
    if epoch % args["iter_check"] == 0: # and epoch != 0:
        # tb eval
        eval(epoch)
        

# %% after training
res = eval(args['n_epoch'])
ev_loss, ev_loss_component, ev_epred_metric = res

if args['tensorboard']:
    tb.add_hparams(args, {'loss': ev_loss, 'xu': ev_loss_component['xu'], 'xv': ev_loss_component['xv'], 
            'xe': ev_loss_component['xe'], 'e': ev_loss_component['e'],
            'acc': ev_epred_metric['acc'], 'f1': ev_epred_metric['f1'],
            'prec': ev_epred_metric['prec'], 'rec': ev_epred_metric['rec'] })

print()
print(args)

