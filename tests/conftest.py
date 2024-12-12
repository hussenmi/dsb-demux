import os
import shutil
import pytest
from sklearn.datasets import make_blobs
from hto_dnd.data import generate_mock_hto_data

@pytest.fixture(scope='module')
def mock_hto_data(request):
    """Generate clustered HTO data."""

    # Parameters
    n_cells = request.param.get("n_cells", 1000)
    n_htos = request.param.get("n_htos", 3)
    noise_level = request.param.get("noise_level", 0.5)

    # Generate data
    mock = generate_mock_hto_data(n_cells=n_cells, n_htos=n_htos, noise_level=noise_level)

    # Save data
    path = "temp"
    path_filtered = path + "/filtered_data.h5ad"
    path_raw = path + "/raw_data.h5ad"
    path_denoised = path + "/out/denoised_data.h5ad"
    os.makedirs(path, exist_ok=True)
    mock["filtered"].write_h5ad(path_filtered)
    mock["raw"].write_h5ad(path_raw)

    # add paths
    mock["path_filtered"] = path_filtered
    mock["path_raw"] = path_raw
    mock["path_denoised"] = path_denoised

    # Yield data
    yield mock

    # clean up after use
    if os.path.exists(path):
        shutil.rmtree(path)
