import pytest
import numpy as np
import anndata as ad
from scipy import sparse
from hto_dnd.dsb_algorithm import remove_batch_effect, dsb

@pytest.fixture
def test_datasets():
    """
    Fixture to create test datasets: filtered and raw.
    """
    # TODO: Add one computed dataset for all tests, not just test_dsb.py
    # Simulate filtered data (cells x proteins)
    filtered_data = np.array([[100, 150, 200], [120, 130, 110], [180, 170, 160]])

    # Simulate raw data with filtered data plus empty droplets
    raw_data = np.array([[100, 150, 200], [120, 130, 110], [180, 170, 160],
                         [10, 15, 20], [12, 13, 11], [18, 17, 16]])  # Empty droplets added

    # Create AnnData objects
    adata_filtered = ad.AnnData(X=filtered_data,
                                obs={'cell_id': ['cell1', 'cell2', 'cell3']},
                                var={'protein_id': ['protein1', 'protein2', 'protein3']})
    adata_raw = ad.AnnData(X=raw_data,
                           obs={'cell_id': ['cell1', 'cell2', 'cell3', 'empty1', 'empty2', 'empty3']},
                           var={'protein_id': ['protein1', 'protein2', 'protein3']})

    return adata_filtered, adata_raw

def test_remove_batch_effect():
    """
    Test remove_batch_effect function.
    """
    x = np.array([[2, 4, 6], [1, 3, 5]])
    covariates = np.array([1, 2])
    design = np.array([[1], [1]])

    corrected_matrix = remove_batch_effect(x, covariates=covariates, design=design)

    expected_corrected_matrix = np.array([[3., 5., 7.], [3., 5., 7.]])
    np.testing.assert_array_almost_equal(corrected_matrix, expected_corrected_matrix)

def test_dsb_basic(test_datasets):
    """
    Test basic functionality of _dsb function.
    """
    # TODO: Preprocess ones and run multiple tests against the results. Currently,
    # the same test is run multiple times.
    adata_filtered, adata_raw = test_datasets

    # Run the _dsb function
    adata_result = dsb(adata_filtered, adata_raw, pseudocount=1, denoise_counts=False)

def test_dsb_denoising(test_datasets):
    """
    Test dsb function with denoising.
    """
    adata_filtered_original, adata_raw = test_datasets

    # Run the dsb function with denoising
    adata_result_denoised = dsb(
        adata_filtered_original,
        adata_raw,
        pseudocount=1,
        add_key_normalise = "normalised",
        add_key_denoise="denoised",
        denoise_counts=True,
        inplace=False,
    )

    # Verify that the normalised and denoised layer is added and has expected dimensions
    assert "normalised" in adata_result_denoised.layers
    assert "denoised" in adata_result_denoised.layers
    assert adata_result_denoised.layers["normalised"].shape == adata_filtered_original.X.shape
    assert adata_result_denoised.layers["denoised"].shape == adata_filtered_original.X.shape

    # Check that denoised values are different from non-denoised values
    assert not np.array_equal(adata_result_denoised.layers["denoised"], adata_result_no_denoise.layers["normalised"]), \
        "Denoised and non-denoised results should be different"

    # Check that the denoised values have lower variance
    denoised_variance = np.var(adata_result_denoised.layers["normalised"])
    non_denoised_variance = np.var(adata_result_denoised.layers["denoised"])
    assert denoised_variance < non_denoised_variance, \
        "Denoised data should have lower variance than non-denoised data"

def test_dsb_sparse_input(test_datasets):
    """
    Test dsb function with sparse input matrices.
    """
    adata_filtered, adata_raw = test_datasets

    # Convert to sparse matrices
    adata_filtered.X = sparse.csr_matrix(adata_filtered.X)
    adata_raw.X = sparse.csr_matrix(adata_raw.X)

    # Run the dsb function
    adata_result = dsb(
        adata_filtered,
        adata_raw,
        pseudocount=1,
        add_key_normalise = "normalised",
        add_key_denoise="denoised",
        denoise_counts=False
    )

    # Verify that the denoised layer is added and is a dense array
    assert "normalised" in adata_result.layers
    assert "denoised" in adata_result.layers
    assert isinstance(adata_result.layers["normalised"], np.ndarray)
    assert isinstance(adata_result.layers["denoised"], np.ndarray)