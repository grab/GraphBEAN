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
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df = pd.read_csv(f"data/movies.csv")

    df["summary_char_len"] = df["summary"].astype("str").apply(len)
    df["text_char_len"] = df["text"].astype("str").apply(len)
    df["helpfulness"] = (
        df["helpfulness_numerator"] / df["helpfulness_denominator"]
    ).fillna(0)

    df = df.sort_values(["product_id", "user_id", "time"])
    dfu = df.groupby(["product_id", "user_id"], as_index=False).last()

    df_product = dfu.groupby("product_id", as_index=False).agg(
        user_count=("user_id", "count"),
        helpful_num_mean=("helpfulness_numerator", "mean"),
        helpful_num_sum=("helpfulness_numerator", "sum"),
        helpful_mean=("helpfulness", "mean"),
        helpful_sum=("helpfulness", "sum"),
        score_mean=("score", "mean"),
        score_sum=("score", "sum"),
        summary_len_mean=("summary_char_len", "mean"),
        summary_len_sum=("summary_char_len", "sum"),
        text_len_mean=("text_char_len", "mean"),
        text_len_sum=("text_char_len", "sum"),
    )

    df_user = dfu.groupby("user_id", as_index=False).agg(
        product_count=("product_id", "count"),
        helpful_num_mean=("helpfulness_numerator", "mean"),
        helpful_num_sum=("helpfulness_numerator", "sum"),
        helpful_mean=("helpfulness", "mean"),
        helpful_sum=("helpfulness", "sum"),
        score_mean=("score", "mean"),
        score_sum=("score", "sum"),
        summary_len_mean=("summary_char_len", "mean"),
        summary_len_sum=("summary_char_len", "sum"),
        text_len_mean=("text_char_len", "mean"),
        text_len_sum=("text_char_len", "sum"),
    )

    df_user.to_csv(f"data/movies-user.csv")
    df_product.to_csv(f"data/movies-product.csv")

    sentences = dfu["text"].astype("str").to_numpy()
    embeddings = model.encode(sentences)

    np.save(f"data/movies-embeddings.npy", embeddings)
    dfu[["product_id", "user_id"]].to_csv(f"data/movies-ids.csv")


def create_graph():

    df_user = pd.read_csv("data/movies-user.csv")
    df_product = pd.read_csv("data/movies-product.csv")
    df_review_id = pd.read_csv("data/movies-ids.csv")
    embeddings = np.load("data/movies-embeddings.npy")

    df_user["uid"] = df_user.index
    df_product["pid"] = df_product.index

    df_user_id = df_user[["user_id", "uid"]]
    df_product_id = df_product[["product_id", "pid"]]

    cols = [f"v{i}" for i in range(embeddings.shape[1])]
    df_review = pd.concat(
        [df_review_id, pd.DataFrame(embeddings, columns=cols)], axis=1
    )

    df_review_2 = df_review.merge(
        df_user_id,
        on="user_id",
    ).merge(df_product_id, on="product_id")
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
    parser.add_argument("--name", type=str, default="movies_anomaly", help="name")
    parser.add_argument("--n-graph", type=int, default=2, help="n graph")

    args = vars(parser.parse_args())

    prepare_data()
    graph = create_graph()
    store_graph("movies-graph", graph)
    # graph = torch.load(f'storage/movies-graph.pt')
    print(graph)

    graph_anomaly_list = []
    for i in range(args["n_graph"]):
        print(f"GRAPH ANOMALY {i} >>>>>>>>>>>>>>")
        graph_multi_dense = inject_random_block_anomaly(
            graph, num_group=100, num_nodes_range=(1, 20)
        )
        graph_anomaly_list.append(graph_multi_dense)
        print()

    dataset = {"args": args, "graph": graph, "graph_anomaly_list": graph_anomaly_list}

    store_graph(args["name"], dataset)


if __name__ == "__main__":
    synth_random()
