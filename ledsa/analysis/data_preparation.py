import numpy as np
import pandas as pd

from ..core.file_handling import extend_hdf, read_hdf


def apply_color_correction(cc_matrix: np.ndarray, on='sum_col_val', channels=(0, 1, 2)) -> None:
    """
    Apply color correction on channel values based on a provided color correction matrix.

    :param cc_matrix: Color correction matrix to apply.
    :type cc_matrix: np.ndarray
    :param on: The reference parameter for color correction, defaults to 'sum_col_val'.
    :type on: str, optional
    :param channels: The channels to consider for color correction, defaults to (0, 1, 2).
    :type channels: tuple, optional
    """
    cc_matrix_inv = np.linalg.inv(cc_matrix)
    quantity = on
    fit_params_list = []
    for channel in channels:
        fit_parameters = read_hdf(channel)[quantity]
        fit_params_list.append(fit_parameters)
    raw_val_array = pd.concat(fit_params_list, axis=1)
    cc_val_array = np.dot(cc_matrix_inv, raw_val_array.T).T
    cc_val_array = cc_val_array.astype(np.int16)
    for channel in channels:
        extend_hdf(channel, quantity + '_cc', cc_val_array[:, channel])