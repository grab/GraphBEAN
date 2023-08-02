# Interaction-Focused Anomaly Detection on Bipartite Node-and-Edge-Attributed Graphs

This repository contains the experimental source code of the [*Interaction-Focused Anomaly Detection on Bipartite Node-and-Edge-Attributed Graphs*](https://engineering.grab.com/graph-anomaly-model) paper presented at the [International Joint Conference on Neural Networks (IJCNN) 2023](https://2023.ijcnn.org/). 

Authors: [Rizal Fathony](mailto:rizal.fathony@grab.com), [Jenn Ng](mailto:jenn.ng@grab.com), and [Jia Chen](mailto:jia.chen@grab.com).

## Abstract

Many anomaly detection applications naturally produce datasets that can be represented as bipartite graphs (user–interaction–item graphs). These graph datasets are usually supplied with rich information on both the entities (nodes) and the interactions (edges). Unfortunately, previous graph neural network anomaly models are unable to fully capture the rich information and produce high-performing detections on these graphs, as they mostly focus on homogeneous graphs and node attributes only. To overcome the problem, we propose a new graph anomaly detection model that focuses on the rich interactions in bipartite graphs. Specifically, our model takes a bipartite node-and-edge-attributed graph and produces anomaly scores for each of its edges and then for each of its bipartite nodes. We design our model as an autoencoder-type model with a customized encoder and decoder to facilitate the compression of node features, edge features, and graph structure into node-level latent representations. The reconstruction errors of each edge and node are then leveraged to spot the anomalies. Our network architecture is scalable, enabling large real-world applications. Finally, we demonstrate that our method significantly outperforms previous anomaly detection methods in the experiments.

## Setup

1. Install the required packages using:
    ```
    pip install -r requirements.txt
    ```
2. Download the datasets.

    - `wikipedia` and `reddit`:
        ```
        wget -P data/ http://snap.stanford.edu/jodie/wikipedia.csv
        wget -P data/ http://snap.stanford.edu/jodie/reddit.csv
        ```

    - `finefoods`:  Download from [here](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?select=Reviews.csv) to `data` folder. Rename the file to `finefoods.csv`.

    - `movies`:  Download from [here](https://snap.stanford.edu/data/web-Movies.html). Extract and rename the file to `movies.txt`. Generate the `.csv` file by running `python extract_movies.py`.


## Construct Graph Datasets

We construct the graph datasets by loading the csv and construct PyG graph data. We then inject anomalies into the dataset. For each dataset, please run:
- `wikipedia` dataset: `python data_wikipedia.py`
- `reddit` dataset: `python data_reddit.py`

- `finefoods-large` dataset: `python data_finefoods.py`
- `finefoods-small` dataset: `python data_finefoods_small.py`
- `movies-large` dataset: `python data_movies.py`
- `movies-small` dataset: `python data_movies_small.py`

`Note`: for `finefoods` and `movies`, we use `sentence-transformer` to generate features from the review text. Running the graph construction on a machine with GPU support is recommended. The size of `finefoods` and `movies` is also quite large. Therefore, a machine with large memory size is required (60GB or 120GB). 

The script will convert the csv files into PyG graph format, and constrcut 10 different copies of the graph by injecting random anomalies into the graph via `anomaly_insert.py`. Each graph instance will have different sets of anomalies. 

## Run Experiment

To run the experiments, please execute the corresponding file for each model. 

1. `GrapBEAN`: 
    ```
    python train_full_experiment.py --name wikipedia_anomaly --id 0
    ```

1. `GrapBEAN` with neighborhood sampling: 
    ```
    python train_sample_experiment.py --name wikipedia_anomaly --id 0 --batch-size 128
    ```

1. `IsolationForest`: 
    ```
    python isoforest_experiment.py --name wikipedia_anomaly --id 0
    ```

1. `DOMINANT`: 
    ```
    python dominant_experiment.py --name wikipedia_anomaly --id 0
    ```

1. `AnomalyDAE`: 
    ```
    python anomalydae_experiment.py --name wikipedia_anomaly --id 0
    ```

1. `AdONE`: 
    ```
    python adone_experiment.py --name wikipedia_anomaly --id 0
    ```

The argument `--name` indicates which dataset we want the model run on, with the format of `{dataset_name}_anomaly`. Additional arguments are also available depending on the models.

- Arguments for **all** models.
    ```
    --name              : dataset name
    --id                : which instance of anomaly injected graph [0-9]
    ```
- Arguments for `DOMINANT`, `AnomalyDAE`, `AdONE`, and `GraphBEAN`.
    ```
    --n-epoch           : number of epoch in the training [default: 50]
    --lr                : learning rate [default: 1e-2]
    ```
- Arguments for `DOMINANT` and `AnomalyDAE`.
    ```
    --alpha             : balance parameter [default: 0.8]
    ```
- Arguments for `GraphBEAN` (full and sample training).
    ```
    --eta                     : structure decoder loss weight [default: 0.2]
    --score-agg               : aggregation method for node anomaly score
                                (max or mean) [default: max]      
    --scheduler-milestones    : milestones for learning scheduler [default: []]            
    ```
- Arguments for `GraphBEAN` (sample training).
    ```
    --batch-size              : number of target nodes in one batch [default: 2048]
    --num-neighbors-u         : number of neighbors sampled for node u [default: 10]
    --num-neighbors-v         : number of neighbors sampled for node v [default: 10]
    --num-workers             : number of workers in dataloader [default: 0]       
                                suggestion: set it as the number of available cores  
    ```

Running the experiments on a machine with GPU support is recommended for all models except IsolationForest.

## License

This repository is licenced under the [MIT License](LICENSE).

## Citation

If you use this repository for academic purpose, please cite the following paper:

```
Rizal Fathony, Jenn Ng, Jia Chen. "Interaction-Focused Anomaly Detection on Bipartite Node-and-Edge-Attributed Graphs." In "International Joint Conference on Neural Networks (IJCNN) 2023". IEEE, 2023.
```