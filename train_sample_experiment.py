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
from models.net_sample import GraphBEANSampled
from models.sampler import BipartiteNeighborSampler
from models.sampler import EdgePredictionSampler
from models.loss import reconstruction_loss
from models.score import compute_anomaly_score, edge_prediction_metric

from utils.sum_dict import dict_addto, dict_div
from utils.seed import seed_all

# %% args

parser = argparse.ArgumentParser(description='GraphBEAN')
parser.add_argument("--name", type=str, default="finefoods_anomaly",
                    help="name")
parser.add_argument("--key", type=str, default="graph_anomaly_list",
                    help="key to the data")
parser.add_argument("--id", type=int, default=0,
                    help="id to the data")
parser.add_argument("--batch-size", type=int, default=2048,
                    help="batch size")
parser.add_argument("--num-neighbors-u", type=int, default=10,
                    help="number of neighbors for node u in sampling")   
parser.add_argument("--num-neighbors-v", type=int, default=10,
                    help="number of neighbors for node v in sampling")    
parser.add_argument("--n-epoch", type=int, default=50,
                    help="number of epoch")    
parser.add_argument("--scheduler-milestones", nargs='+', type=int, default=[20, 35],
                    help="scheduler milestone")     
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--score-agg', type=str, default='max', help='aggregation for node anomaly score')
   
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
    "structure_loss_weight" : 0.2,
    "structure_loss_weight_anomaly_score" : 0.2,
    "iter_check" : 10,
    "seed" : 0,
    "num_workers" : 16,
    "neg_sampler_mult" : 3,
    'k_check' : 15,
    'tensorboard' : False,
    'progress_bar' : False
}

args = {**args1, **args2}

seed_all(args['seed'])

data_dir = '/workspace/results/'


# %% params
batch_size = args["batch_size"]

# %% data
data = load_graph(args['name'], args['key'], args['id'])
print(data)

u_ch = data.xu.shape[1]
v_ch = data.xv.shape[1]
e_ch = data.xe.shape[1]

print(f"Data dimension: U node = {data.xu.shape}; V node = {data.xv.shape}; E edge = {data.xe.shape}; \n")

# %% model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphBEANSampled(
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

xu, xv = data.xu, data.xv
xe, adj = data.xe, data.adj
yu, yv, ye = data.yu, data.yv, data.ye

# sampler
train_loader = BipartiteNeighborSampler(adj, n_layers=4, base='v', batch_size=batch_size, drop_last=True,
                                n_other_node=-1, num_neighbors_u=args["num_neighbors_u"], num_neighbors_v=args["num_neighbors_v"],
                                num_workers=args['num_workers'], shuffle=True)  

print(args)
print()

# %% train
def train(epoch, check_counter):

    model.train()

    n_batch = len(train_loader)
    if args['progress_bar']:
        pbar = tqdm(total=n_batch, leave=False)
        pbar.set_description(f'#{epoch:3d}')

    total_loss = 0
    total_epred_metric = {'acc': 0.0, 'prec': 0.0, 'rec': 0.0, 'f1': 0.0}
    total_loss_component = {'xu': 0.0, 'xv': 0.0, 'xe': 0.0, 'e': 0.0, 'total': 0.0}
    num_update = 0

    for batch_size, indices, adjacencies, e_flags in train_loader:

        # print(f"# u nodes: {len(indices[0])} | # v nodes: {len(indices[1])} | # edges: {len(indices[2])}")

        adjacencies = [adj.to(device) for adj in adjacencies]
        e_flags = [fl.to(device) for fl in e_flags]
        u_id, v_id, e_id = indices

        # sample
        xu_sample = xu[u_id].to(device)
        xv_sample = xv[v_id].to(device)
        xe_sample = xe[e_id].to(device)

        # edge pred samples
        target_adj = adjacencies[-1].adj_e.adj
        edge_pred_sampler = EdgePredictionSampler(target_adj, mult=args["neg_sampler_mult"])
        edge_pred_samples = edge_pred_sampler.sample().to(device)

        optimizer.zero_grad()

        # start = time.time()
        out = model(xu=xu_sample, xv=xv_sample, xe=xe_sample,
                    bean_adjs=adjacencies, e_flags=e_flags,
                    edge_pred_samples=edge_pred_samples)
        # print(f"training : {time.time() - start} s")

        last_adj_e = adjacencies[-1].adj_e
        xu_target = xu[last_adj_e.u_id].to(device)
        xv_target = xv[last_adj_e.v_id].to(device)
        xe_target = xe[last_adj_e.e_id].to(device)

        loss, loss_component = reconstruction_loss(xu=xu_target, xv=xv_target, xe=xe_target, 
                    adj=last_adj_e.adj, edge_pred_samples=edge_pred_samples, out=out, 
                    xe_loss_weight=args["xe_loss_weight"], structure_loss_weight=args["structure_loss_weight"])

        loss.backward()
        optimizer.step()

        epred_metric = edge_prediction_metric(edge_pred_samples, out['eprob'])

        total_loss += float(loss)
        total_epred_metric = dict_addto(total_epred_metric, epred_metric)
        total_loss_component = dict_addto(total_loss_component, loss_component)
        num_update += 1

        if args['progress_bar']:
            pbar.update(1)
            pbar.set_postfix({'loss' : float(loss), 'ep acc': epred_metric['acc'], 'ep f1': epred_metric['f1']})

        if num_update == args["k_check"]:
            loss = total_loss / num_update
            loss_component = dict_div(total_loss_component, num_update)
            epred_metric = dict_div(total_epred_metric, num_update)

            # tensorboard
            if args['tensorboard']:
                tb.add_scalar('loss', loss, check_counter)
                tb.add_scalar('loss_xu', loss_component['xu'], check_counter)
                tb.add_scalar('loss_xv', loss_component['xv'], check_counter)
                tb.add_scalar('loss_xe', loss_component['xe'], check_counter)
                tb.add_scalar('loss_e', loss_component['e'], check_counter)

                tb.add_scalar('epred_acc', epred_metric['acc'], check_counter)
                tb.add_scalar('epred_f1', epred_metric['f1'], check_counter)
                tb.add_scalar('epred_prec', epred_metric['prec'], check_counter)
                tb.add_scalar('epred_rec', epred_metric['rec'], check_counter)

            check_counter += 1

            total_loss = 0
            total_epred_metric = {'acc': 0.0, 'prec': 0.0, 'rec': 0.0, 'f1': 0.0}
            total_loss_component = {'xu': 0.0, 'xv': 0.0, 'xe': 0.0, 'e': 0.0, 'total': 0.0}
            num_update = 0

    if args['progress_bar']:
        pbar.close()
    scheduler.step()

    return loss, loss_component, epred_metric, check_counter

# %% evaluate and store
def eval(epoch):

    model.eval()

    start = time.time()

    # negative sampling
    edge_pred_sampler = EdgePredictionSampler(adj, mult=args["neg_sampler_mult"])
    edge_pred_samples = edge_pred_sampler.sample()

    with torch.no_grad():
        
        out = model.inference(xu, xv, xe, adj, edge_pred_samples, 
                            batch_sizes=(2**13, 2**13, 2**13),
                            device=device, progress_bar=args['progress_bar'])
        
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
    torch.save(model_stored, os.path.join(data_dir, f"graphbean-{args['name']}-{args['id']}-model.th"))
    torch.save(output_stored, os.path.join(data_dir, f"graphbean-{args['name']}-{args['id']}-output.th"))

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

# eval(0)

for epoch in range(args["n_epoch"]):

    start = time.time()
    loss, loss_component, epred_metric, check_counter = train(epoch, check_counter)
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

