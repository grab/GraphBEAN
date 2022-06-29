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

# %%


def standardize(features: np.ndarray) -> np.ndarray:
    scaler = preprocessing.StandardScaler()
    z = scaler.fit_transform(features)
    return z


def sample_data():
    df_user = pd.read_csv(f"data/finefoods-user.csv")
    df_product = pd.read_csv(f"data/finefoods-product.csv")
    df_review = pd.read_csv(f"data/finefoods-review.csv")

    pc = np.log10(df_user["product_count"].to_numpy()) + 1
    user_weight = pc / pc.sum()

    uc = np.log10(df_product["user_count"].to_numpy()) + 1
    product_weight = uc / uc.sum()

    user_nums = np.random.choice(df_user.shape[0], 24000, replace=False, p=user_weight)
    user_ids = df_user["UserId"][user_nums]

    product_nums = np.random.choice(
        df_product.shape[0], 12000, replace=False, p=product_weight
    )
    product_ids = df_product["ProductId"][product_nums]

    df_review_chosen = df_review[
        df_review["ProductId"].isin(product_ids) & df_review["UserId"].isin(user_ids)
    ].iloc[:, 1:]
    df_user_chosen = df_user[
        df_user["UserId"].isin(df_review_chosen["UserId"].unique())
    ].iloc[:, 1:]
    df_product_chosen = df_product[
        df_product["ProductId"].isin(df_review_chosen["ProductId"].unique())
    ].iloc[:, 1:]

    df_user_chosen.to_csv(f"data/finefoods-small-user.csv")
    df_product_chosen.to_csv(f"data/finefoods-small-product.csv")
    df_review_chosen.to_csv(f"data/finefoods-small-review.csv")


def create_graph():

    df_user = pd.read_csv("data/finefoods-small-user.csv")
    df_product = pd.read_csv("data/finefoods-small-product.csv")
    df_review = pd.read_csv("data/finefoods-small-review.csv")

    df_user["uid"] = df_user.index
    df_product["pid"] = df_product.index

    df_user_id = df_user[["UserId", "uid"]]
    df_product_id = df_product[["ProductId", "pid"]]

    df_review_2 = df_review.merge(
        df_user_id,
        on="UserId",
    ).merge(df_product_id, on="ProductId")
    df_review_2 = df_review_2.sort_values(["uid", "pid"])

    uid = torch.tensor(df_review_2["uid"].to_numpy())
    pid = torch.tensor(df_review_2["pid"].to_numpy())

    adj = SparseTensor(row=uid, col=pid)
    edge_attr = torch.tensor(standardize(df_review_2.iloc[:, 3:-2].to_numpy())).float()

    user_attr = torch.tensor(standardize(df_user.iloc[:, 2:-1].to_numpy())).float()
    product_attr = torch.tensor(
        standardize(df_product.iloc[:, 2:-1].to_numpy())
    ).float()

    data = BipartiteData(adj, xu=user_attr, xv=product_attr, xe=edge_attr)

    return data


def store_graph(name: str, dataset):
    torch.save(dataset, f"storage/{name}.pt")


def load_graph(name: str, key: str, id=None):
    if id == None:
        data = torch.load(f"storage/{name}.pt")
        return data[key]
    else:
        data = torch.load(f"storage/{name}.pt")
        return data[key][id]


def synth_random():
    # generate nd store data
    import argparse

    parser = argparse.ArgumentParser(description="GraphBEAN")
    parser.add_argument(
        "--name", type=str, default="finefoods-small_anomaly", help="name"
    )
    parser.add_argument("--n-graph", type=int, default=10, help="n graph")

    args = vars(parser.parse_args())

    sample_data()
    graph = create_graph()
    store_graph("finefoods-small-graph", graph)
    # graph = torch.load(f'storage/finefoods-small-graph.pt')

    graph_anomaly_list = []
    for i in range(args["n_graph"]):
        print(f"GRAPH ANOMALY {i} >>>>>>>>>>>>>>")
        graph_multi_dense = inject_random_block_anomaly(
            graph, num_group=20, num_nodes_range=(1, 12)
        )
        graph_anomaly_list.append(graph_multi_dense)
        print()

    dataset = {"args": args, "graph": graph, "graph_anomaly_list": graph_anomaly_list}

    store_graph(args["name"], dataset)


if __name__ == "__main__":
    synth_random()
