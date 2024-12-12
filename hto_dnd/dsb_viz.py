import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compare_distributions(raw_data, dsb_data, output_path):
    n_htos = raw_data.shape[1]
    fig, axes = plt.subplots(n_htos, 2, figsize=(15, 4*n_htos))

    for i, hto in enumerate(raw_data.columns):
        # Raw data
        raw_data_hto = raw_data[hto]
        dsb_data_hto = dsb_data[hto]

        # Calculate x-axis limits for raw data
        # raw_xlim = np.percentile(raw_data_hto, [0, 99])
        log_raw_data_hto = np.log(raw_data_hto+1)

        sns.histplot(log_raw_data_hto, ax=axes[i, 0], kde=True)
        axes[i, 0].set_title(f'{hto} - Raw, log scale')
        # axes[i, 0].set_xlim(raw_xlim)

        # DSB data
        # Calculate x-axis limits for DSB data
        # dsb_xlim = np.percentile(dsb_data_hto, [0, 99.5])

        sns.histplot(dsb_data_hto, ax=axes[i, 1], kde=True)
        axes[i, 1].set_title(f'{hto} - DSB')
        # axes[i, 1].set_xlim(dsb_xlim)

        # Add text with min, max, and median values
        axes[i, 0].text(0.05, 0.95, f"Min: {log_raw_data_hto.min():.2f}\nMax: {log_raw_data_hto.max():.2f}\nMedian: {log_raw_data_hto.median():.2f}",
                        transform=axes[i, 0].transAxes, verticalalignment='top')
        axes[i, 1].text(0.05, 0.95, f"Min: {dsb_data_hto.min():.2f}\nMax: {dsb_data_hto.max():.2f}\nMedian: {dsb_data_hto.median():.2f}",
                        transform=axes[i, 1].transAxes, verticalalignment='top')

    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path)
    plt.close()

def create_visualization(adata, output_path, layer=None):
    """Creates and saves a visualization comparing pre and DSB-normalized data.

    Args:
        adata (AnnData): An AnnData object containing the data.
        output_path (str): The file path where the visualization will be saved.
    Returns:
        None
    """
    # Extract raw and DSB-normalized data
    raw_data = pd.DataFrame(adata.X, columns=adata.var_names)
    if layer is None:
        dsb_data = pd.DataFrame(adata.X, columns=adata.var_names)
    else:
        dsb_data = pd.DataFrame(adata.layers[layer], columns=adata.var_names)

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create and save the plot
    compare_distributions(raw_data, dsb_data, output_path)