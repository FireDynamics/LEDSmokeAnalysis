import numpy as np
import pandas as pd

from ..core.file_handling import extend_hdf, read_hdf


def add_normalized_parameter(parameter: str, channels=(0, 1, 2)):
    """
    normalizes one parameter of the binary and adds the new column 'normalized_{parameter}' to it
    """
    for channel in channels:
        bin_parameters = read_hdf(channel)
        if f"normalized_{parameter}" not in bin_parameters.columns:
            average = _calculate_average_fitpar_without_smoke(bin_parameters, parameter, channel)
            normalized_par = bin_parameters[parameter].div(average)
            extend_hdf(channel, "normalized_" + parameter, normalized_par)


def apply_color_correction(cc_matrix: np.ndarray, on='sum_col_val', channels=(0, 1, 2)) -> None:
    """ Apply color correction on channel values based on color correction matrix.
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


def _calculate_average_fitpar_without_smoke(bin_parameters: pd.DataFrame, par: str, channel: int, num_of_imgs=20) -> pd.DataFrame:
    idx = pd.IndexSlice
    pars_without_smoke = bin_parameters.loc[idx[1:num_of_imgs, :]]
    return pars_without_smoke[par].mean(0, level='led_id')
