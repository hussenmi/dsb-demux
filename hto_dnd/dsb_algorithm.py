## In thif file, we implement DSB on raw data, where we have empty droplets. We the use the bimodal distribution of the HTOs to set a threshold for each HTO. This threshold is used to classify a cell into negative or non negative. If a cell is classified as non negative by more than one HTo, it's considered a doublet.

import os
import numpy as np
import scipy
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
import anndata as ad
import pandas as pd
from pandas.api.types import is_integer_dtype

from .logging import get_logger
from .dsb_viz import create_visualization

from line_profiler import profile
@profile
def remove_batch_effect(x, covariates=None, design=None):
    """Remove batch effects from a given matrix using linear regression.

    This function removes technical noise (batch effects) from the input matrix by fitting
    a linear model with the provided covariates and design matrix. The correction is then
    subtracted from the original matrix.

    Args:
        x (ndarray): Input matrix from which batch effects will be removed.
        covariates (ndarray, optional): Matrix of technical covariates (e.g. GMM means)
            used to model batch effects.
        design (ndarray, optional): Design matrix for the linear regression model. If not
            provided, uses a column vector of ones.

    Returns:
        ndarray: Matrix with batch effects removed.
    """
    x = np.asarray(x)

    if design is None:
        design = np.ones((x.shape[0], 1))
    else:
        design = np.asarray(design)

    # Process covariates (in our case, this is the GMM means)
    if covariates is not None:
        covariates = np.asarray(covariates).reshape(-1, 1)

    # Combine design and covariates
    X_combined = np.column_stack([design, covariates])

    # Fit linear model
    model = LinearRegression(fit_intercept=False)
    model.fit(X_combined, x)

    # Extract coefficients related to batch effects
    beta = model.coef_[:, design.shape[1] :]
    # beta = model.coef_

    # Broadcast the multiplication. here beta is the coefficient of the regression and covariates is the baclground means. their multiplication is just the prediction of how much technical noise there is and then after we predict that, we subtract it from x (the normalized matrix) to get the corrected matrix
    correction = covariates @ beta.T

    # Subtract the correction from x to remove the batch effect
    x_corrected = x - correction

    # Store metadata
    meta = {
        "coefs": model.coef_,
    }

    return x_corrected, meta


@profile
def dsb(
    adata_filtered: ad.AnnData,
    adata_raw: ad.AnnData,
    pseudocount: int = 10,
    denoise_counts: bool = True,
    add_key_normalise: str = None,
    add_key_dnd: str = None,
    inplace: bool = False,
    path_adata_out: str = None,
    create_viz: bool = False,
    verbose: int = 1,
) -> ad.AnnData:
    """Custom implementation of the DSB (Denoised and Scaled by Background) algorithm.

    This function implements an adapted version of the DSB algorithm, which normalizes protein
    expression data using empty droplets as background reference and optionally performs
    technical noise removal.
    Args:
        adata_filtered (AnnData): Filtered AnnData object containing protein expression data
            for cells passing QC.
        adata_raw (AnnData): Raw AnnData object containing protein expression data for all
            droplets including empty ones.
        pseudocount (int, optional): Value added to expression counts before log transformation.
            Defaults to 10.
        denoise_counts (bool, optional): Whether to perform technical noise removal using
            Gaussian Mixture Models. Defaults to True.
        add_key_normalise (str, optional): Key to store the normalized data in the AnnData object. Default is None.
        add_key_dnd (str, optional): Key to store the normalised and denoised data in the AnnData object. Default is None.
        inplace (bool, optional): Flag indicating whether to modify the input AnnData object. Default is False.
        path_adata_out (str, optional): Path to save the output AnnData object. Default is None.
        create_viz (bool, optional): Flag indicating whether to create a visualization plot. Default is False.
        verbose (int, optional): Verbosity level. Default is 1.

    Returns:
        AnnData: The input adata_filtered object with an additional layer 'dsb_normalized'
            containing the normalized protein expression values.

        The normalized data is stored in the layers attribute of the returned AnnData object
        under 'dsb_normalized'.
    """

    # assertions
    if inplace:
        raise NotImplementedError("Inplace operation is not supported.")
    assert is_integer_dtype(adata_filtered.X), "Filtered counts must be integers."
    assert is_integer_dtype(adata_raw.X), "Raw counts must be integers."

    # Get logger
    logger = get_logger(level=verbose)
    logger.info("Starting DSB normalization...")

    # Setup
    adata = adata_filtered.copy()

    # Create cell_protein_matrix
    cell_protein_matrix = adata.X  # .T
    if scipy.sparse.issparse(cell_protein_matrix):
        cell_protein_matrix = cell_protein_matrix.toarray()

    # Identify barcodes that are in adata_raw but not in adata_filtered
    # Convert to sets
    raw_barcodes = set(adata_raw.obs_names)
    filtered_barcodes = set(adata.obs_names)

    # Find the difference
    empty_barcodes = list(raw_barcodes - filtered_barcodes)

    # Get the empty droplets from adata_raw
    empty_drop_matrix = adata_raw[empty_barcodes, :].X  # .T
    if scipy.sparse.issparse(empty_drop_matrix):
        empty_drop_matrix = empty_drop_matrix.toarray()

    adt = np.array(cell_protein_matrix)
    adtu = np.array(empty_drop_matrix)

    # Log transform both matrices
    adt_log = np.log(adt + pseudocount)
    adtu_log = np.log(adtu + pseudocount)

    # Calculate mean and sd of log-transformed empty droplets for each protein
    mu_empty = np.mean(adtu_log, axis=0)
    sd_empty = np.std(adtu_log, axis=0)

    # Normalize the cell protein matrix
    normalized_matrix = (adt_log - mu_empty) / sd_empty

    # Store meta information
    meta_normalise = {
        "normalise": {
            "pseudocount": pseudocount,
            "mean_empty": mu_empty,
            "sd_empty": sd_empty,
        }
    }
    adata.uns["dnd"] = {
        **adata.uns.get("dnd", {}),
        **meta_normalise
    }

    # Checkpoint
    if add_key_normalise is not None:
        adata.layers[add_key_normalise] = normalized_matrix
        logger.info(f"Normalized matrix stored in adata.layers['{add_key_normalise}']")
    else:
        adata.X = normalized_matrix
        logger.info("DSB normalization completed.")

    if not denoise_counts:
        return adata

    # Step 2: Technical noise removal
    logger.info("Removing technical noise...")

    # Apply a 2-component GMM for each cell and get the first component mean
    n_cells, n_proteins = normalized_matrix.shape

    def _get_background(x):
        gmm = GaussianMixture(n_components=2, random_state=0).fit(x.reshape(-1, 1))
        return min(gmm.means_)[0]

    noise_vector = np.array([
        _get_background(normalized_matrix[i, :])
        for i in range(n_cells)
    ])

    norm_adt, meta_batch_model = remove_batch_effect(normalized_matrix, covariates=noise_vector)
    logger.info("Technical noise removal completed.")

    # Store meta information
    meta_dnd = {
        "dnd": {
            "background_means": noise_vector,
            "batch_model": meta_batch_model,
        }
    }
    adata.uns["dnd"] = {
        **adata.uns.get("dnd", {}),
        **meta_dnd
    }

    # After computing norm_adt, update the AnnData object
    if add_key_dnd is not None:
        adata.layers[add_key_dnd] = norm_adt
        logger.info(f"DND matrix stored in adata.layers['{add_key_dnd}']")
    else:
        adata.X = norm_adt
        logger.info("DND matrix stored in adata.X")

    # Save outputs (try catch to return the adata object even if saving fails)
    try:
        path_viz = os.path.join(os.getcwd(), "dsb_viz.png")

        if path_adata_out is not None:
            adata.write_h5ad(path_adata_out)
            path_viz = os.path.join(
                os.path.dirname(path_adata_out),
                os.path.basename(path_adata_out).split(".")[0] + "_dsb_viz.png",
            )
            logger.info(f"AnnData object saved to '{path_adata_out}'")

        if create_viz:
            create_visualization(adata, path_viz)
            logger.info(f"Visualization plot saved to '{path_viz}'")
    except Exception as e:
        logger.error(f"Failed to save outputs: '{e}'")

    return adata
