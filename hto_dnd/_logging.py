import logging
from pprint import pformat
import anndata as ad

def _log_parameters(logger : logging.Logger, params : dict):
    """Log parameters to the logger."""
    params = {k: v.shape if isinstance(v, ad.AnnData) else v for k, v in params.items()}
    params_str = pformat(params, indent=4)
    logger.debug(f"Parameters:\n{params_str}")

# setup logging to the console in the format:
# <timestamp> <log level> - <message>
def get_logger(logger, level : int = 1):
    # Set logger level
    log = logging.getLogger(logger)
    level = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }.get(level, logging.INFO)
    log.setLevel(level)

    # Clear existing handlers
    if log.hasHandlers():
        log.handlers.clear()

    formatter = logging.Formatter("%(asctime)s HTO-DND %(name)s | %(levelname)s: %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    log.addHandler(ch)

    # Add log_parameters function to the logger
    log.log_parameters = lambda params: _log_parameters(log, params)
    return log
