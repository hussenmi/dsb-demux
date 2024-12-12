from math import floor
import numpy as np
from itertools import chain
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


# OUT OF THE BOX
def _apply_gmm_to_row(x, func):
    """Apply a function to each row of the input matrix using Gaussian Mixture Models."""
    metas = []
    n_cells = x.shape[0]
    results = [func(x[i, :]) for i in range(n_cells)]
    noise_vector = np.array([r[0] for r in results])
    metas = [r[1] for r in results]
    meta = {"debug": {"metas": metas}}
    return noise_vector, meta


def _get_background_gmm(normalized_matrix, **kwargs):
    """Fit a Gaussian Mixture Model to the input data and return the mean of the first component."""
    def _get_background(x):
        gmm = GaussianMixture(n_components=2, random_state=0, **kwargs).fit(x.reshape(-1, 1))
        return min(gmm.means_)[0], {"debug": {"model": gmm}}
    return _apply_gmm_to_row(normalized_matrix, _get_background)


def _get_background_kmeans(normalized_matrix, **kwargs):
    """Fit a KMeans model to the input data and return the mean of the first cluster."""
    def _get_background(x):
        kmeans = KMeans(n_clusters=2, random_state=0, **kwargs).fit(x.reshape(-1, 1))
        return min(kmeans.cluster_centers_)[0], {"model": kmeans}
    return _apply_gmm_to_row(normalized_matrix, _get_background)


# CUSTOM KMEANS
def _converge(matrix, center_lower, center_upper, n_iter):
    """Converge to the background noise vectors."""
    # converge
    for i in range(n_iter):
        row_mid = (center_lower + center_upper) / 2
        assignment = matrix > row_mid[:, np.newaxis]
        center_lower = np.sum(np.where(~assignment, matrix, 0), axis=1) / np.sum(~assignment, axis=1)
        center_upper = np.sum(np.where(assignment, matrix, 0), axis=1) / np.sum(assignment, axis=1)

    # intertia
    inertia = np.sum(np.where(~assignment, (matrix - center_lower[:, np.newaxis])**2, 0), axis=1)
    inertia += np.sum(np.where(assignment, (matrix - center_upper[:, np.newaxis])**2, 0), axis=1)
    return center_lower, center_upper, inertia

def _init_strategies(normalized_matrix, method, *args, **kwargs):
    """Return iterator of initialisation strategies.

    Supported methods:
    - 'rank': rank-based initialisation. args: k (int) - which rank to use from top and bottom.
    - 'rank_lower': rank-based initialisation. args: k (int) - which rank to use from bottom.
    - '1vall': Use max as top and mean of the rest as bottom.
    """

    if method == "rank":
        k = kwargs.get("k", 0)
        assert k >= 0, "Rank must be greater than 0."
        assert k < floor(normalized_matrix.shape[1] / 2), "Rank must be less than half the number of HTOs."
        center_lower_init = np.partition(normalized_matrix, k, axis=1)[:, k]
        center_upper_init = np.partition(normalized_matrix, -k, axis=1)[:, -k]
        yield center_lower_init, center_upper_init

    elif method == "rank_lower":
        k = kwargs.get("k", 0)
        assert k >= 0, "Rank must be greater or equal to 0."
        assert k < normalized_matrix.shape[1] - 1, "Rank must be less than the number of HTOs - 1."
        center_lower_init = np.partition(normalized_matrix, k, axis=1)[:, k]
        center_upper_init = np.max(normalized_matrix, axis=1)
        yield center_lower_init, center_upper_init

    elif method == "1vall":
        center_upper_init = np.max(normalized_matrix, axis=1)
        center_lower_init = np.mean(np.partition(normalized_matrix, -1, axis=1)[:, :-1], axis=1)
        yield center_lower_init, center_upper_init

    else:
        raise ValueError(f"Method '{method}' not supported.")

def _all_inits(matrix):
    """Generate all supported inits - feasible for lower dimensions.
    """
    from math import floor
    m = matrix.shape[1]
    inits = [{"method": "1vall"}]
    for i in range(m - 1):
        inits.append({"method": "rank_lower", "k": i})
    for i in range(floor(m / 2)):
        inits.append({"method": "rank", "k": i})
    return inits


def _get_background_kmeans_fast(matrix, n_iter=5, inits=None):
    """Simple heuristics for fast, row-wise KMeans with 2 clusters.

    1. Initialize clusters using 'inits' strategies.
    2. Iterate through initialisations:
        a. Converge with simple KMeans algorithm.
        b. Update cluster centers using cluster with lowest inertia.

    Note that '1vall' is computationally efficient, while preserving almost exact results
    compared to 'KMeans' and 'GMM' for HTO data.

    Example:
    ```
    matrix = np.random.normal(0, 1, (100, 10))
    background = _get_background_kmeans_fast(matrix)
    ```
    """

    # init
    assert isinstance(matrix, np.ndarray), "Matrix must be a numpy array."
    if inits is None:
        inits = _all_inits(matrix)
    intertia_min = np.repeat(np.inf, matrix.shape[0])
    center_lower_min = np.zeros(matrix.shape[0])

    # get iterator
    inits = chain(*[_init_strategies(matrix, **init) for init in inits])

    for center_inits in inits:
        # converge
        center_lower, center_upper, inertia = _converge(matrix, *center_inits, n_iter=n_iter)
        # update if better
        center_lower_min = np.where(inertia < intertia_min, center_lower, center_lower_min)
        center_upper_min = np.where(inertia < intertia_min, center_upper, center_upper)
        intertia_min = np.minimum(inertia, intertia_min)

    # store data
    assignment = matrix > ((center_lower_min + center_upper_min) / 2).reshape(-1, 1)
    meta = {
        "debug": {
            "intertia": intertia_min,
            "center_lower": center_lower_min,
            "center_upper": center_upper_min,
            "assignment": assignment,
        }
    }

    # return mean of the lower cluster
    return center_lower_min, meta


def assert_background(method, **kwargs):
    supported_methods = ["kmeans_fast", "kmeans", "gmm"]
    assert method in supported_methods, f"Method '{method}' not supported. Choose from: {', '.join(supported_methods)}"

def estimate_background(
    matrix,
    method="kmeans_fast",
    **kwargs,
):
    if method=="kmeans_fast":
        return _get_background_kmeans_fast(matrix, **kwargs)
    elif method=="kmeans":
        return _get_background_kmeans(matrix, **kwargs)
    elif method=="gmm":
        return _get_background_gmm(matrix, **kwargs)
    else:
        raise ValueError(f"Method '{method}' not supported.")
