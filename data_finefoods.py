# Copyright 2021 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import torch
from torch_sparse.tensor import SparseTensor

import numpy as np
from anomaly_insert import inject_random_block_anomaly

from models.data import BipartiteData

import torch
from sklearn import preprocessing

import pandas as pd

from sentence_transformers import SentenceTransformer

# %% 

def standardize(features: np.ndarray) -> np.ndarray:
    scaler = preprocessing.StandardScaler()
    z = scaler.fit_transform(features)
    return z

def prepare_data():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = pd.read_csv(f'data/finefoods.csv')

    df['SummaryCharLen'] = df['Summary'].astype('str').apply(len)
    df['TextCharLen'] = df['Text'].astype('str').apply(len)
    df["Helpfulness"] = (df["HelpfulnessNumerator"] / df["HelpfulnessDenominator"]).fillna(0)

    df = df.iloc[:,1:].sort_values(['ProductId', 'UserId', 'Time'])
    dfu = df.groupby(["ProductId", 'UserId'], as_index=False).last()

    df_product = dfu.groupby("ProductId", as_index=False).agg(
        user_count=('UserId', 'count'), helpful_num_mean=('HelpfulnessNumerator', 'mean'), helpful_num_sum=('HelpfulnessNumerator', 'sum'),
        helpful_mean=('Helpfulness', 'mean'), helpful_sum=('Helpfulness', 'sum'), score_mean=('Score', 'mean'), score_sum=('Score', 'sum'),
        summary_len_mean=('SummaryCharLen', 'mean'), summary_len_sum=('SummaryCharLen', 'sum'), text_len_mean=('TextCharLen', 'mean'), text_len_sum=('TextCharLen', 'sum')
        )
    
    df_user = dfu.groupby("UserId", as_index=False).agg(
        product_count=('ProductId', 'count'), helpful_num_mean=('HelpfulnessNumerator', 'mean'), helpful_num_sum=('HelpfulnessNumerator', 'sum'),
        helpful_mean=('Helpfulness', 'mean'), helpful_sum=('Helpfulness', 'sum'), score_mean=('Score', 'mean'), score_sum=('Score', 'sum'),
        summary_len_mean=('SummaryCharLen', 'mean'), summary_len_sum=('SummaryCharLen', 'sum'), text_len_mean=('TextCharLen', 'mean'), text_len_sum=('TextCharLen', 'sum')
        )
    
    df_user.to_csv(f'data/finefoods-user.csv')
    df_product.to_csv(f'data/finefoods-product.csv')

    sentences = dfu['Text'].astype('str').to_numpy()
    embeddings = model.encode(sentences)
    cols = [f'v{i}' for i in range(embeddings.shape[1])]
    df_review = pd.concat([dfu[['ProductId', 'UserId']], pd.DataFrame(embeddings, columns=cols)], axis=1)

    df_review.to_csv(f'data/finefoods-review.csv')
    
def create_graph():

    df_user = pd.read_csv('data/finefoods-user.csv')
    df_product = pd.read_csv('data/finefoods-product.csv')
    df_review = pd.read_csv('data/finefoods-review.csv')

    df_user["uid"] = df_user.index
    df_product["pid"] = df_product.index

    df_user_id = df_user[["UserId", "uid"]]
    df_product_id = df_product[['ProductId', 'pid']]

    df_review_2 = df_review.merge(
        df_user_id,
        on = "UserId",
    ).merge(
        df_product_id,
        on = "ProductId"
    )
    df_review_2 = df_review_2.sort_values(['uid', 'pid'])

    uid = torch.tensor(df_review_2['uid'].to_numpy())
    pid = torch.tensor(df_review_2['pid'].to_numpy())

    adj = SparseTensor(row=uid, col=pid)
    edge_attr = torch.tensor(standardize(df_review_2.iloc[:,3:-2].to_numpy())).float()

    user_attr = torch.tensor(standardize(df_user.iloc[:,2:-1].to_numpy())).float()
    product_attr = torch.tensor(standardize(df_product.iloc[:,2:-1].to_numpy())).float()

    data = BipartiteData(adj, xu=user_attr, xv=product_attr, xe=edge_attr)

    return data

def store_graph(name: str, dataset):
    torch.save(dataset, f'storage/{name}.pt')

def load_graph(name: str, key: str, id=None):
    if id == None:
        data = torch.load(f'storage/{name}.pt')
        return data[key]
    else:
        data = torch.load(f'storage/{name}.pt')
        return data[key][id]


def synth_random():
    # generate nd store data
    import argparse

    parser = argparse.ArgumentParser(description='GraphBEAN')
    parser.add_argument("--name", type=str, default="finefoods_anomaly",
                        help="name")
    parser.add_argument("--n-graph", type=int, default=5,
                        help="n graph")

    args = vars(parser.parse_args())

    prepare_data()
    graph = create_graph()
    store_graph('finefoods-graph', graph)
    # graph = torch.load(f'storage/finefoods-graph.pt')

    graph_anomaly_list = []
    for i in range(args['n_graph']):
        print(f"GRAPH ANOMALY {i} >>>>>>>>>>>>>>")
        graph_multi_dense = inject_random_block_anomaly(graph, num_group=100, num_nodes_range=(1,20))
        graph_anomaly_list.append(graph_multi_dense)
        print()

    dataset = {'args' : args, 'graph' : graph, 'graph_anomaly_list' : graph_anomaly_list}

    store_graph(args['name'], dataset)

    
if __name__ == "__main__":
    synth_random()
