
class AnnDataFormatError(Exception):
    def __init__(self, message, key, *args, **kwargs):
        self.message = self.get_message(key, *args, **kwargs)
        super().__init__(message)

    def get_message(self, key, *args, **kwargs):
        _func = {
            "adata_raw_missing_cells": self._adata_raw_missing_cells,
            "adata_filtered_too_few_cells": self._adata_filtered_too_few_cells,
        }.get(key, self._default)
        return _func(*args, **kwargs)

    def _default(self, *args, **kwargs):
        return f"Unknown error occurred. Args: {args}, kwargs: {kwargs}"

    def _adata_raw_missing_cells(self, empty_barcodes):
        return f"Too few empty droplets detected: '{len(empty_barcodes)}'. " \
               f"Make sure that 'adata_raw' contains cells that are not present in 'adata_filtered'."

    def _adata_filtered_too_few_cells(self, n_cells):
        return f"Too few cells in 'adata_filtered': '{n_cells}'. "