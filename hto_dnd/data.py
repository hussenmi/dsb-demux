import numpy as np
import anndata as ad
from sklearn.datasets import make_blobs


def generate_mock_hto_data(n_cells=1000, n_htos=3, noise_level=0.5, seed=42):
    """Generate clustered HTO data.

    Creates the following labels:
    - "filtered"
        - 70% Singlet: High in one HTO cluster
        - 20% Doublet: High in two HTO clusters
        - 10% Negative: Low in all HTO clusters
    - "raw"
        - +100% Empty: Very low in all HTO clusters

    Returns:
        Dictionary with the following keys:
        - 'filtered': AnnData object with filtered data
        - 'path_filtered': Path to the filtered data
        - 'raw': Raw data
        - 'path_raw': Path to the raw data
        - 'path_denoised': Path to the output data
        - 'filtered_labels': Labels for the filtered data
        - 'raw_labels': Labels for the raw data
        - 'filtered_cell_types': Cell types for the filtered data
        - 'raw_cell_types': Cell types for the raw data
    """

    # Set seed
    np.random.seed(seed)
    assert n_htos >=2, "Number of HTOs must be at least 2"

    # Define cluster centers
    # - off-diagonal is random around 0-5
    # - diagonal is random around 10
    hto_centers = np.random.uniform(0, 5, (n_htos, n_htos))
    hto_centers[np.diag_indices(n_htos)] = np.random.uniform(5, 15, n_htos)

    negative_center = np.ones(n_htos) * 2  # All HTOs low
    empty_center = np.ones(n_htos) * 0.5  # All HTOs very low

    # Calculate number of cells for each type
    n_singlets = int(n_cells * 0.7)
    n_doublets = int(n_cells * 0.2)
    n_negatives = n_cells - n_singlets - n_doublets
    n_empty = int(n_cells)  # 100% additional empty droplets

    # Generate singlets
    singlets, singlet_labels = make_blobs(n_samples=n_singlets, n_features=n_htos,
                                          centers=hto_centers, cluster_std=3.0)

    # Generate doublets
    doublets = []
    doublet_labels = []
    for _ in range(n_doublets):
        hto1, hto2 = np.random.choice(n_htos, 2, replace=False)
        doublet = (hto_centers[hto1] + hto_centers[hto2]) * np.random.uniform(0.5, 1)
        doublet += np.random.normal(0, 0.2, n_htos)
        doublets.append(doublet)
        doublet_labels.append("Doublet")
    doublets = np.array(doublets)

    # Generate negatives
    negatives, _ = make_blobs(n_samples=n_negatives, n_features=n_htos,
                              centers=[negative_center], cluster_std=0.5)

    # Generate empty droplets
    empty, _ = make_blobs(n_samples=n_empty, n_features=n_htos,
                          centers=[empty_center], cluster_std=0.2)

    # Combine all data
    raw_data = np.vstack((singlets, doublets, negatives, empty))

    # Create labels
    singlet_labels = [f'{i}' for i in singlet_labels]
    negative_labels = ['Negative'] * n_negatives
    empty_labels = ['Empty'] * n_empty
    raw_labels = np.array(singlet_labels + doublet_labels + negative_labels + empty_labels)
    raw_cell_types = np.array(['Singlet'] * n_singlets + doublet_labels + negative_labels + empty_labels)
    subset_filterd = raw_cell_types != 'Empty'

    # Ensure non-negative values
    raw_data = np.maximum(raw_data, 0)

    # Add non-negative noise
    raw_data += np.random.lognormal(0, noise_level, raw_data.shape)

    # To integer
    raw_data = np.round(raw_data).astype(int)

    # Create anndata
    cell_ids = [f'cell{i+1}' for i in range(raw_data.shape[0])]
    hto_ids = [f'HTO_{i+1}' for i in range(n_htos)]

    raw_adata = ad.AnnData(
        X=raw_data,
        obs={'cell_id': cell_ids,
             'cell_type': raw_cell_types,
             'hto_classification': raw_labels},
        var={'hto_id': hto_ids}
    )
    filtered_adata = raw_adata[subset_filterd].copy()

    # return
    return {
        "filtered": filtered_adata,
        "raw": raw_adata,
        "filtered_labels": raw_labels[subset_filterd],
        "raw_labels": raw_labels,
        "filtered_cell_types": raw_cell_types[subset_filterd],
        "raw_cell_types": raw_cell_types
    }
