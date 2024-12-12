import pytest
import numpy as np
import pandas as pd
import anndata as ad
import yaml
from sklearn.datasets import make_blobs
from hto_dnd.demux_dsb import cluster_and_evaluate, demux

def test_demux(mock_dsb_denoised_adata, tmp_path):
    """
    Test the demux function for demultiplexing using different methods.
    Parameters:
        mock_dsb_denoised_adata: Mock AnnData object used for testing.
        tmp_path: Temporary directory for storing test files.
    Methods tested:
        - 'kmeans'
        - 'gmm'
    Assertions:
        - The result is an instance of ad.AnnData.
        - 'hashID' and 'Doublet_Info' columns exist in result.obs.
        - 'metrics' exists in result.uns.
        - Classifications contain valid values ('Negative', 'Doublet', or start with 'HTO_').
        - Metrics exist for each HTO
        - Overall accuracy compared to true labels is at least 0.8
    """
    temp_file = tmp_path / "mock_dsb_denoised_adata.h5ad"
    mock_dsb_denoised_adata.write(temp_file)

    true_labels = mock_dsb_denoised_adata.obs['true_label']


    for method in ['kmeans', 'gmm']:
        adata_filtered = ad.read_h5ad(temp_file)
        result = demux(adata_filtered, method=method)

        assert isinstance(result, ad.AnnData)
        assert 'hashID' in result.obs.columns
        assert 'Doublet_Info' in result.obs.columns
        assert 'metrics' in result.uns["dnd"]["denoise"]

        # Check if classifications exist
        classifications = result.obs['hashID'].value_counts()
        assert len(classifications) > 0
        assert all(classification in ['Negative', 'Doublet'] or classification.startswith('HTO_')
                   for classification in classifications.index)

        # Check if metrics exist for each HTO
        assert all(f'HTO_{i}' in result.uns['metrics'] for i in range(3))

        # Check overall accuracy
        predicted_labels = result.obs['hashID']
        overall_accuracy = np.mean(predicted_labels == true_labels)
        assert overall_accuracy > 0.8, f"Overall accuracy for {method} is only {overall_accuracy:.2f}"


def test_consistent_classification(mock_dsb_denoised_adata, tmp_path):
    """
    This test verifies that running the demux function twice with the same
    input data produces identical results in terms of the 'hashID' observed in the
    output AnnData object
    Parameters:
        mock_dsb_denoised_adata: A mock AnnData object used for testing.
        tmp_path: A temporary directory path for storing the mock data file.
    Assertions:
        Asserts that the 'hashID' series from the results of two consecutive calls
        to demux are equal.
    """
    temp_file = tmp_path / "mock_dsb_denoised_adata.h5ad"
    mock_dsb_denoised_adata.write(temp_file)

    adata_filtered = ad.read_h5ad(temp_file)
    result1 = demux(adata_filtered, method='kmeans')
    result2 = demux(adata_filtered, method='kmeans')

    pd.testing.assert_series_equal(result1.obs['hashID'], result2.obs['hashID'])


@pytest.mark.parametrize("method", ["kmeans", "gmm", "otsu"])
def test_cluster_and_evaluate(mock_dsb_denoised_adata, method):
    """
    Test the clustering and evaluation of HTO data.
    Checks the following:
    1. The number of unique labels is exactly 2.
    2. The identified positive cluster is either 0 or 1.
    3. For "kmeans":
        - The metrics dictionary contains 'silhouette_score' and 'davies_bouldin_index'.
        - The silhouette score is positive
        - The Davies-Bouldin index is positive.
    4. For the "gmm" method:
        - The metrics dictionary contains 'bic' and 'log_likelihood'.
        - The log-likelihood is negative, as expected.
    5. For the "otsu" method:
        - The metrics dictionary contains 'threshold', 'inter_class_variance', and 'entropy'.
        - The threshold is a float greater than 0.
    Parameters:
         mock_dsb_denoised_adata: An AnnData object containing the denoised data.
         method: A string indicating the clustering method to use ("kmeans" or "gmm").
    """
    adata = mock_dsb_denoised_adata
    X = adata.X

    for i in range(X.shape[1]):
        hto_data = X[:, i].reshape(-1, 1)
        if method == "otsu":
            labels, threshold, metrics = cluster_and_evaluate(hto_data, method=method)
        else:
            labels, positive_cluster, metrics = cluster_and_evaluate(hto_data, method=method)
            assert positive_cluster in [0, 1]

        assert len(np.unique(labels)) == 2

        if method == "kmeans":
            assert 'silhouette_score' in metrics
            assert 'davies_bouldin_index' in metrics
            assert metrics['silhouette_score'] > 0
            assert metrics['davies_bouldin_index'] > 0
        elif method == "gmm":
            assert 'bic' in metrics
            assert 'log_likelihood' in metrics
            assert metrics['log_likelihood'] < 0
        elif method == "otsu":
            assert 'threshold' in metrics
            assert 'inter_class_variance' in metrics
            assert 'entropy' in metrics
            assert isinstance(metrics['threshold'], float)
            assert threshold > 0

def test_well_separated_clusters_kmeans():
    """
    Test the clustering of well-separated clusters using the KMeans algorithm.
    Generates synthetic data with two distinct clusters using the
    make_blobs function. It then applies the clustering and evaluation
    method to ensure that the following conditions are met:
    - The number of clusters is equal to 2.
    - silhouette score is greater than 0.75, indicating well-separated clusters.
    - Davies-Bouldin index is less than 0.5, suggesting low similarity between clusters.
    """
    X, _ = make_blobs(n_samples=300, centers=[(0, 0), (10, 10)], cluster_std=0.5, random_state=42)
    labels, positive_cluster, metrics = cluster_and_evaluate(X, method='kmeans')

    assert len(np.unique(labels)) == 2
    assert metrics['silhouette_score'] > 0.75  # Well-separated clusters should have high silhouette score
    assert metrics['davies_bouldin_index'] < 0.5  # Well-separated clusters should have low Davies-Bouldin index


def test_overlapping_clusters_kmeans():
    """
    Test the clustering of overlapping clusters using the KMeans algorithm.
    Generates synthetic data with overlapping clusters using the
    make_blobs function. It then applies the clustering and evaluation
    method to ensure that the following conditions are met:
    - The number of clusters should be 2.
    - silhouette score should be less than 0.5.
    - Davies-Bouldin index should be greater than 0.5.
    """
    X, _ = make_blobs(n_samples=300, centers=[(0, 0), (1, 1)], cluster_std=1.0, random_state=42)
    labels, positive_cluster, metrics = cluster_and_evaluate(X, method='kmeans')

    assert len(np.unique(labels)) == 2
    assert metrics['silhouette_score'] < 0.5  # Overlapping clusters should have lower silhouette score
    assert metrics['davies_bouldin_index'] > 0.5  # Overlapping clusters should have higher Davies-Bouldin index

def test_invalid_method():
    """
    Test the behavior of the cluster_and_evaluate function when an invalid method is provided.
    """
    X = np.random.rand(100, 2)
    with pytest.raises(ValueError):
        cluster_and_evaluate(X, method='invalid_method')