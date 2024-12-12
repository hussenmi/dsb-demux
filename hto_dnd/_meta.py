

SUPPORTED_STEPS = ["normalise", "denoise", "demux", "paths"]

def init_meta(adata):
    """Initialize the meta data for the AnnData object.

    DNDs metadata is structured as follows:
    adata.uns["dnd"] = {
        "normalise": {
            "method": "method_name",
            "params": {"param1": value1, "param2": value2, ...},
        },
    """
    adata.uns["dnd"] = {
        **{step: {} for step in SUPPORTED_STEPS},
        **adata.uns.get("dnd", {}),
    }
    return adata

def add_meta(adata, step, params={}, **kwargs):
    """Add metadata to the AnnData object.

    Args:
        adata: AnnData object.
        step: The step of the DND pipeline (should be in "normalise", "denoise", "demux", "paths")
        params: Parameters used in the step.
        kwargs: Additional metadata to add.

    Example:
    ```
    adata.uns["dnd"] = {
        "normalise": {},
        "denoise": {},
        "demux": {},
    }
    adata = add_meta(adata, step="normalise", params={"method": "method_name"}, key1="value1", key2="value2")
    # >> adata.uns["dnd"]["normalise"]["key1"] == "value1"
    # >> adata.uns["dnd"]["normalise"]["params"]["method"] == "method_name"
    ```
    """

    assert step in ["normalise", "denoise", "demux", "paths"], f"Invalid step: '{step}'"
    meta_new = {
        step: {
            "params": params,
            **kwargs,
        },
    }
    adata.uns["dnd"] = {
        **adata.uns["dnd"],
        **meta_new,
    }
    return adata
