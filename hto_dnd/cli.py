"""
Can be used in this way:

# Perform DSB normalization
python cli.py dsb --adata-filtered-in path/to/filtered_data.h5ad --adata-raw-in path/to/raw_data.h5ad --adata-out path/to/output_data.h5ad --create-viz

# Perform demultiplexing
python cli.py demux --dsb-denoised-adata-dir path/to/normalised_data.h5ad --method kmeans --output-path path/to/demultiplexed_data.h5ad

# Perform DSB normalization and demultiplexing
python cli.py dsb_and_demux --adata_filtered_dir path/to/filtered_data.h5ad --adata_raw_dir path/to/raw_data.h5ad --output-path path/to/output_data.h5ad
"""


import argparse
import anndata as ad
from hto_dnd.dsb_algorithm import dsb
from hto_dnd.demux_dsb import demux
from hto_dnd.dsb_and_demux import dsb_and_demux

def parse_dsb_arguments():
    """Parse command line arguments for DSB analysis.

    This function sets up and parses command line arguments needed for DSB (Diagnostic
    Strand-seq Browser) analysis, including paths for input/output AnnData files and
    visualization options.

    Returns:
        argparse.Namespace: An object containing all the parsed arguments with the following attributes:
            path_adata_filtered_in (str): Path to filtered input AnnData file (.h5ad)
            path_adata_raw_in (str): Path to raw input AnnData file (.h5ad)
            path_adata_out (str): Path to output AnnData file (.h5ad)
            create_viz (bool): Whether to create visualization plots (default: False)
            pseudocount (float): Pseudocount value used in the formula to avoid taking log of 0 (default: 10)
            denoise_counts (bool): Flag indicating whether to perform correction to cell variation using the background values (default: True)
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--adata-filtered-in",
        action="store",
        dest="path_adata_filtered_in",
        help="path to filtered input AnnData (.h5ad)",
        required=True,
    )

    parser.add_argument(
        "--adata-raw-in",
        action="store",
        dest="path_adata_raw_in",
        help="path to raw input AnnData (.h5ad)",
        required=True,
    )

    parser.add_argument(
        "--adata-out",
        action="store",
        dest="path_adata_out",
        help="path to output AnnData (.h5ad). Visualization (if created) will be saved in the same directory.",
        required=True,
    )

    parser.add_argument(
        "--create-viz",
        action="store_true",
        dest="create_viz",
        help="create visualization plot for the impact of DSB (default: False)",
        default=False,
    )

    parser.add_argument(
        "--pseudocount",
        action="store",
        dest="pseudocount",
        help="pseudocount value used in the formula to avoid taking log of 0 (default: 10)",
        type=float,
        default=10,
    )

    parser.add_argument(
        "--denoise-counts",
        action="store_true",
        dest="denoise_counts",
        help="flag indicating whether to perform correction to cell variation using the background values (default: True)",
        default=True,
    )

    parser.add_argument(
        "--add-key-normalise",
        action="store",
        dest="add_key_normalise",
        help="key to store the normalized data in the AnnData object",
        default="normalised",
    )

    parser.add_argument(
        "--add-key-denoise",
        action="store",
        dest="add_key_denoise",
        help="key to store the normalised and denoised data in the AnnData object",
        default="denoised",
    )

    # parse arguments
    params = parser.parse_args()
    return params

def parse_demux_arguments():
    """Parse command line arguments for HTO demultiplexing.
    This function sets up and parses command line arguments for the HTO demultiplexing process.
    It handles paths for input data, method selection, layer specification, and output location.
    Returns:
        argparse.Namespace: A namespace object containing the following attributes:
            path_dsb_denoised_adata_dir (str): Path to directory containing DSB denoised AnnData files
            method (str): Clustering method to use for demultiplexing (default: 'kmeans')
            layer (str): Data layer to use for demultiplexing (default: 'dnd')
            output_path (str): Path where the output AnnData file will be saved
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dsb-denoised-adata-dir",
        action="store",
        dest="path_dsb_denoised_adata_dir",
        help="path to DSB denoised anndata directory",
        required=True,
    )
    parser.add_argument(
        "--method",
        action="store",
        dest="method",
        help="method used to cluster when demultiplexing",
        default="kmeans",
    )

    parser.add_argument(
        "--layer",
        action="store",
        dest="layer",
        help="layer to use for demultiplexing",
        default="denoised",
    )

    parser.add_argument(
        "--output-path",
        action="store",
        dest="output_path",
        help="path to an output adata file",
        required=True,
    )

    # parse arguments
    params = parser.parse_args()

    return params

def parse_dsb_and_demux_arguments():
    """Parse command line arguments for DSB and demux operations.
    This function sets up and parses command line arguments for processing filtered
    and raw AnnData files and specifying an output path.
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            path_adata_filtered_dir (str): Path to filtered input AnnData directory
            path_adata_raw_dir (str): Path to raw input AnnData directory
            output_path (str): Path where the output AnnData file will be saved
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--adata_filtered_dir",
        action="store",
        dest="path_adata_filtered_dir",
        help="path to filtered input AnnData directory",
        required=True,
    )

    parser.add_argument(
        "--adata_raw_dir",
        action="store",
        dest="path_adata_raw_dir",
        help="path to raw input AnnData directory",
        required=True,
    )

    parser.add_argument(
        "--output-path",
        action="store",
        dest="output_path",
        help="path to an output adata file",
        required=True,
    )

    # parse arguments
    params = parser.parse_args()

    return params

def main():
    """CLI interface for HTO demultiplexing operations.

    This function serves as the main entry point for the command-line interface,
    providing three main functionalities: DSB normalization, demultiplexing, and
    a combined DSB-demultiplexing operation.
    The CLI supports three commands:
        - dsb: Performs DSB normalization on HTO data
        - demux: Performs demultiplexing on DSB-normalized data
        - dsb_and_demux: Combines DSB normalization and demultiplexing in one step
    Returns:
        None
    Command-line Arguments:
        For 'dsb' command:
            --adata-filtered-in: Path to filtered AnnData file
            --adata-raw-in: Path to raw AnnData file
            --adata-out: Output path for processed AnnData
            --create-viz: Flag to create visualizations
            --pseudocount: Pseudocount value for normalization (default: 10)
            --denoise-counts: Flag to denoise counts (default: True)
        For 'demux' command:
            --dsb-denoised-adata-dir: Directory with DSB-normalized AnnData
            --method: Demultiplexing method (default: "kmeans")
            --layer: Layer name in AnnData (default: "dnd")
            --output-path: Path for output file
        For 'dsb_and_demux' command:
            --adata_filtered_dir: Directory with filtered AnnData
            --adata_raw_dir: Directory with raw AnnData
            --output-path: Path for output file
    Example:
        $ python script.py dsb --adata-filtered-in filtered.h5ad --adata-raw-in raw.h5ad --adata-out output.h5ad
        $ python script.py demux --dsb-denoised-adata-dir processed.h5ad --output-path result.h5ad
        $ python script.py dsb_and_demux --adata_filtered_dir filtered.h5ad --adata_raw_dir raw.h5ad --output-path final.h5ad
    """

    parser = argparse.ArgumentParser(description="HTO DND CLI")
    subparsers = parser.add_subparsers(dest="command")

    dsb_parser = subparsers.add_parser("dsb")
    dsb_parser.add_argument("--adata-filtered-in", required=True)
    dsb_parser.add_argument("--adata-raw-in", required=True)
    dsb_parser.add_argument("--adata-out", required=True)
    dsb_parser.add_argument("--create-viz", action="store_true", default=False)
    dsb_parser.add_argument("--pseudocount", type=float, default=10)
    dsb_parser.add_argument("--denoise-counts", action="store_true", default=True)

    demux_parser = subparsers.add_parser("demux")
    demux_parser.add_argument("--dsb-denoised-adata-dir", required=True)
    demux_parser.add_argument("--method", default="kmeans")
    demux_parser.add_argument("--layer", default="dnd")
    demux_parser.add_argument("--output-path", required=True)

    dsb_and_demux_parser = subparsers.add_parser("dsb_and_demux")
    dsb_and_demux_parser.add_argument("--adata_filtered_dir", required=True)
    dsb_and_demux_parser.add_argument("--adata_raw_dir", required=True)
    dsb_and_demux_parser.add_argument("--output-path", required=True)

    args = parser.parse_args()

    if args.command == "dsb":
        adata_filtered = ad.read(args.adata_filtered_in)
        adata_raw = ad.read(args.adata_raw_in)
        dsb(
            adata_filtered=adata_filtered,
            adata_raw=adata_raw,
            path_adata_out=args.adata_out,
            create_viz=args.create_viz,
            pseudocount=args.pseudocount,
            denoise_counts=args.denoise_counts,
        )
    elif args.command == "demux":
        adata_result = demux(
            adata_denoised=ad.read(args.dsb_denoised_adata_dir),
            method=args.method,
            layer=args.layer,
        )
        adata_result.write(args.output_path)
    elif args.command == "dsb_and_demux":
        adata_filtered = ad.read(args.adata_filtered_dir)
        adata_raw = ad.read(args.adata_raw_dir)
        demux_adata = dsb_and_demux(
            adata_filtered=adata_filtered,
            adata_raw=adata_raw,
            path_adata_out=args.output_path,
        )
        demux_adata.write(args.output_path)

if __name__ == "__main__":
    main()