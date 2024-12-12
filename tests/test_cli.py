import subprocess
import os
import numpy as np
from sklearn.datasets import make_blobs
import anndata as ad
import pytest

def generate_clustered_hto_data(n_cells=1000, n_htos=3, noise_level=0.5):
    """Generate clustered HTO data."""
    hto_centers = np.array([[100,10,10], [10,100,10], [10,10,100]])
    negative_center = np.ones(n_htos) * 2
    empty_center = np.ones(n_htos) * 0.5

    n_singlets = int(n_cells * 0.7)
    n_doublets = int(n_cells * 0.2)
    n_negatives = n_cells - n_singlets - n_doublets
    n_empty = int(n_cells * 0.3)

    singlets, singlet_labels = make_blobs(n_samples=n_singlets, n_features=n_htos,
                                          centers=hto_centers, cluster_std=3.0)

    doublets = []
    doublet_labels = []
    for _ in range(n_doublets):
        hto1, hto2 = np.random.choice(n_htos, 2, replace=False)
        doublet = (hto_centers[hto1] + hto_centers[hto2]) * np.random.uniform(0.5, 1)
        doublet += np.random.normal(0, 0.2, n_htos)
        doublets.append(doublet)
        doublet_labels.append("Doublet")
    doublets = np.array(doublets)

    negatives, _ = make_blobs(n_samples=n_negatives, n_features=n_htos,
                              centers=[negative_center], cluster_std=0.5)

    empty, _ = make_blobs(n_samples=n_empty, n_features=n_htos,
                          centers=[empty_center], cluster_std=0.2)

    filtered_data = np.vstack((singlets, doublets, negatives))
    raw_data = np.vstack((filtered_data, empty))

    singlet_labels = [f'{i}' for i in singlet_labels]
    negative_labels = ['Negative'] * n_negatives
    empty_labels = ['Empty'] * n_empty

    filtered_labels = singlet_labels + doublet_labels + negative_labels
    raw_labels = filtered_labels + empty_labels

    filtered_cell_types = ['Singlet'] * n_singlets + ['Doublet'] * n_doublets + ['Negative'] * n_negatives
    raw_cell_types = filtered_cell_types + ['Empty'] * n_empty

    filtered_data = np.maximum(filtered_data, 0)
    raw_data = np.maximum(raw_data, 0)

    filtered_data += np.random.lognormal(0, noise_level, filtered_data.shape)
    raw_data += np.random.lognormal(0, noise_level, raw_data.shape)

    filtered_data = np.round(filtered_data).astype(int)
    raw_data = np.round(raw_data).astype(int)

    return filtered_data, raw_data, filtered_labels, raw_labels, filtered_cell_types, raw_cell_types

def create_anndata(data, labels, cell_types, n_htos):
    """Create AnnData object from the generated data."""
    cell_ids = [f'cell{i+1}' for i in range(len(data))]
    hto_ids = [f'HTO_{i+1}' for i in range(n_htos)]

    adata = ad.AnnData(X=data,
                       obs={'cell_id': cell_ids,
                            'cell_type': cell_types,
                            'hto_classification': labels},
                       var={'hto_id': hto_ids})
    return adata

@pytest.fixture
def test_datasets_with_files(tmp_path):
    """
    Fixture to create test datasets and save them to temporary files.
    """
    # Generate synthetic data
    n_cells = 1000
    n_htos = 3
    filtered_data, raw_data, filtered_labels, raw_labels, filtered_cell_types, raw_cell_types = generate_clustered_hto_data(n_cells, n_htos)

    adata_filtered = create_anndata(filtered_data, filtered_labels, filtered_cell_types, n_htos)
    adata_raw = create_anndata(raw_data, raw_labels, raw_cell_types, n_htos)

    # Save to temporary files
    filtered_path = tmp_path / "filtered.h5ad"
    raw_path = tmp_path / "raw.h5ad"
    adata_filtered.write(filtered_path)
    adata_raw.write(raw_path)

    return str(filtered_path), str(raw_path), adata_filtered, adata_raw

def run_cli_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result

def test_dsb_cli(test_datasets_with_files, tmp_path):
    filtered_path, raw_path, adata_filtered, adata_raw = test_datasets_with_files
    dsb_output_path = tmp_path / "dsb_output.h5ad"
    viz_output_path = tmp_path / "dsb_viz.png"
    pseudocount = 10
    denoise_counts = True

    command = f"python -m hto_dnd.cli dsb --adata-filtered-in {filtered_path} --adata-raw-in {raw_path} --adata-out {dsb_output_path} --create-viz --pseudocount {pseudocount} --denoise-counts"
    result = run_cli_command(command)

    assert result.returncode == 0, f"CLI command failed: {result.stderr}"
    assert os.path.exists(dsb_output_path), "DSB output file not created"
    # assert os.path.exists(viz_output_path), "Visualization file not created"
    # Check if visualization file was created
    viz_filename = os.path.splitext(os.path.basename(dsb_output_path))[0] + "_dsb_viz.png"
    viz_output_path = os.path.join(os.path.dirname(dsb_output_path), viz_filename)
    assert os.path.exists(viz_output_path), f"Visualization file not created at {viz_output_path}"


    adata_dsb = ad.read_h5ad(dsb_output_path)
    assert "denoised" in adata_dsb.layers, "Denoised and normalised layer not found"

def test_demux_cli(test_datasets_with_files, tmp_path):
    filtered_path, raw_path, adata_filtered, adata_raw = test_datasets_with_files
    dsb_output_path = tmp_path / "dsb_output.h5ad"
    demux_output_path = tmp_path / "demux_output.h5ad"

    # Run DSB first
    dsb_command = f"python -m hto_dnd.cli dsb --adata-filtered-in {filtered_path} --adata-raw-in {raw_path} --adata-out {dsb_output_path} --create-viz"
    dsb_result = run_cli_command(dsb_command)
    assert dsb_result.returncode == 0, f"CLI command failed: {dsb_result.stderr}"

    # Run demux
    demux_command = f"python -m hto_dnd.cli demux --dsb-denoised-adata-dir {dsb_output_path} --method kmeans --output-path {demux_output_path}"
    demux_result = run_cli_command(demux_command)
    assert demux_result.returncode == 0, f"CLI command failed: {demux_result.stderr}"

    adata_result = ad.read_h5ad(demux_output_path)
    assert isinstance(adata_result, ad.AnnData), "Demultiplexing result is not an AnnData object"
    assert 'hashID' in adata_result.obs.columns, "hashID column not found in demultiplexing result"
    assert 'Doublet_Info' in adata_result.obs.columns, "Doublet_Info column not found in demultiplexing result"
    assert 'metrics' in adata_result.uns, "Metrics not found in demultiplexing result"

def test_dsb_and_demux_cli(test_datasets_with_files, tmp_path):
    filtered_path, raw_path, adata_filtered, adata_raw = test_datasets_with_files
    dsb_and_demux_output_path = tmp_path / "dsb_and_demux_output.h5ad"

    command = f"python -m hto_dnd.cli dsb_and_demux --adata_filtered_dir {filtered_path} --adata_raw_dir {raw_path} --output-path {dsb_and_demux_output_path}"
    result = run_cli_command(command)

    assert result.returncode == 0, f"CLI command failed: {result.stderr}"
    assert os.path.exists(dsb_and_demux_output_path), "DSB and demux output file not created"

    adata_result = ad.read_h5ad(dsb_and_demux_output_path)
    assert isinstance(adata_result, ad.AnnData), "DSB and demux result is not an AnnData object"
    assert 'hashID' in adata_result.obs.columns, "hashID column not found in DSB and demux result"
    assert 'Doublet_Info' in adata_result.obs.columns, "Doublet_Info column not found in DSB and demux result"
    assert 'metrics' in adata_result.uns, "Metrics not found in DSB and demux result"