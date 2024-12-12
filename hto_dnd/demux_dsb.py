#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from pprint import pformat
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import yaml
from skimage.filters import threshold_otsu
import scipy.stats
import anndata as ad

from ._logging import get_logger
from ._meta import init_meta, add_meta
from line_profiler import profile

def numpy_to_python(obj):
    """Convert NumPy types to native Python types.

    Args:
        obj (object): The object to convert.

    Returns:
        object: The converted object.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(i) for i in obj]
    else:
        return obj

def write_stats(result_df, metrics, output_file="stats.yml"):
    """Write statistics and metrics to a YAML file.

    Args:
        result_df (pd.DataFrame): The result DataFrame containing the hashID and Doublet_Info columns.
        metrics (dict): A dictionary containing the metrics for each HTO.
        output_file (str): The output file path. Default is 'stats.yml'.

    Returns:
        None
    """
    stats = result_df.groupby(by="hashID").size().to_dict()
    stats["Total"] = len(result_df)

    # Convert NumPy values to native Python types
    metrics = numpy_to_python(metrics)

    output_dict = {"stats": stats, "metrics": metrics}

    # Write stats and metrics to the YAML file
    with open(output_file, "wt") as fout:
        yaml.dump(output_dict, fout, sort_keys=False, default_flow_style=False)

def cluster_and_evaluate(data, method="kmeans"):
    """Perform clustering and evaluate it using multiple metrics for K-means or GMM.

    Args:
        data (np.array): The input data used for clustering
        method (str): Clustering method to use. Either 'kmeans' or 'gmm'. Default is 'kmeans'.

    Returns:
        tuple: A tuple containing:
            - np.array: The cluster labels
            - int: The index of the positive cluster
            - dict: A dictionary containing various goodness of fit metrics
    """
    n_clusters = 2
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(data)
        positive_cluster = np.argmax(model.cluster_centers_)

        silhouette = silhouette_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)

        metrics = {"silhouette_score": silhouette, "davies_bouldin_index": davies_bouldin}

    elif method == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        model.fit(data)
        labels = model.predict(data)
        positive_cluster = np.argmax(model.means_)

        bic = model.bic(data)
        log_likelihood = model.score(data) * data.shape[0]

        metrics = {"bic": bic, "log_likelihood": log_likelihood}

        return labels, positive_cluster, metrics

    elif method == "otsu":
        # Ensure data is 1D
        data = data.flatten()

        # Calculate Otsu's threshold
        threshold = threshold_otsu(data)

        # Create binary labels
        labels = (data > threshold).astype(int)

        # Calculate some metrics
        background = data[labels == 0]
        signal = data[labels == 1]

        # Inter-class variance (which Otsu's method maximizes)
        weight1 = np.sum(labels == 0) / len(labels)
        weight2 = np.sum(labels == 1) / len(labels)
        inter_class_variance = weight1 * weight2 * (np.mean(signal) - np.mean(background))**2

        # Calculate entropy of the thresholded image
        hist, _ = np.histogram(labels, bins=2)
        hist_norm = hist / np.sum(hist)
        entropy = scipy.stats.entropy(hist_norm)

        metrics = {
            "threshold": threshold,
            "inter_class_variance": inter_class_variance,
            "entropy": entropy
        }

        return labels, threshold, metrics

    else:
        raise ValueError("Method must be either 'kmeans' or 'gmm'")

    return labels, positive_cluster, metrics

@profile
def demux(
    adata_denoised: ad.AnnData,
    method: str = "kmeans",
    layer: str = None,
    save_stats: bool = False,
    inplace: bool = False,
    verbose: int = 1,
):
    """
    Classify HTOs as singlets (assign to HTO), doublets, or negatives.

    It uses a 2-component K-means, GMM, or Otsu threshold method to categorize cells based on their HTO classifications.

    Args:
        adata_denoised (ad.AnnData): The DSB denoised AnnData object.
        method (str): Clustering method to use. Must be either 'gmm' or 'kmeans'. Default is 'kmeans'.
        layer (str): The layer to use for demultiplexing. Default is 'None'.
        save_stats (bool): Whether to save the statistics to a YAML file. Default is False.
        inplace (bool): Whether to modify the input AnnData object. Default is False.
        verbose (int): Verbosity level. Default is 1.

    Returns:
        AnnData: An AnnData object containing the results of the demultiplexing.
    """

    # debug - print parameters
    logger = get_logger("demux", level=verbose)
    params = {k: v.shape if isinstance(v, ad.AnnData) else v for k, v in locals().items()}
    params_str = pformat(params, indent=4)
    logger.debug(f"Parameters:\n{params_str}")
    logger.info(f"Starting demultiplexing using '{method}'...")

    # assertions
    if inplace:
        raise NotImplementedError("Inplace operation is not supported.")

    # get data
    adata = adata_denoised.copy()
    df_umi = adata.to_df(layer=layer)
    assert all([t == float for t in df_umi.dtypes]), "Denoised data must be float."

    # initialize meta
    adata = init_meta(adata)

    classifications = []
    metrics = {}
    thresholds = {}

    for hto in df_umi.columns:
        data = df_umi[hto].values.reshape(-1, 1)

        # Perform clustering and evaluation in one step
        if method == "otsu":
            labels, threshold, hto_metrics = cluster_and_evaluate(data, method=method)
            thresholds[hto] = threshold
            metrics[hto] = hto_metrics

            # Clasiify the points based on the threshold
            classifications.append(labels)
        else:
            labels, positive_cluster, hto_metrics = cluster_and_evaluate(data, method=method)
            metrics[hto] = hto_metrics

            # Classify the points based on the cluster labels
            classifications.append(
                (labels == positive_cluster).astype(int)
            )

    result_df = pd.DataFrame(
        classifications, index=df_umi.columns, columns=df_umi.index
    ).T

    # Categorize cells based on their HTO classifications
    def categorize_cell(row):
        positive_htos = row[row == 1].index.tolist()
        if len(positive_htos) == 0:
            return "Negative", None
        elif len(positive_htos) == 1:
            return positive_htos[0], None
        else:
            return "Doublet", ",".join(positive_htos)

    result_df["hash_id"], result_df["doublet_info"] = zip(
        *result_df.apply(categorize_cell, axis=1)
    )


    # create an anndata object where the denoised data is the X matrix, the barcodes and features are the obs and var names, add the hashID and Doublet_Info as an obs column, and metrics as an uns
    adata.obs["hash_id"] = result_df["hash_id"]
    adata.obs["doublet_info"] = result_df["doublet_info"]

    # add metadata
    adata = add_meta(
        adata,
        step="demux",
        params=dict(
            method=method,
            layer=layer,
        ),
        metrics=metrics,
        thresholds=thresholds if method == "otsu" else None,
    )

    logger.info(f"Demultiplexing completed. Assignments found as 'hash_id' and 'doublet_info' in adata.obs.")

    if save_stats:
        write_stats(result_df, metrics)

    return adata