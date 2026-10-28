"""Unit tests for the colour corrected channel values.

The corrected values used to be stored as int16, while the sum over a search area of 8 bit
pixels already reaches far beyond that range: a sum_col_val of 72000 was written as 6464, so
the _cc column held wrapped values for every well exposed LED.
"""
import numpy as np
import pandas as pd
import pytest

from ledsa.analysis.data_preparation import apply_color_correction
from ledsa.core.file_handling import read_hdf

SUM_COL_VALS = [300.0, 72000.0, 100000.0]


def write_binaries(tmp_path, monkeypatch, values=SUM_COL_VALS) -> None:
    """Write one analysis binary per channel in an empty working directory.

    :param values: The sum_col_val of every LED of a single image, the same for all channels.
    """
    monkeypatch.chdir(tmp_path)
    for channel in range(3):
        directory = tmp_path / 'analysis' / f'channel{channel}'
        directory.mkdir(parents=True)
        pd.DataFrame({'img_id': 1, 'led_id': range(len(values)), 'sum_col_val': values}).to_hdf(
            directory / 'all_parameters.h5', key='channel_values', format='table')


def corrected_values(channel: int) -> np.ndarray:
    return read_hdf(channel)['sum_col_val_cc'].to_numpy()


def test_the_corrected_values_are_not_truncated(tmp_path, monkeypatch):
    write_binaries(tmp_path, monkeypatch)

    # the identity leaves every channel value as it is, so only the storage is tested
    apply_color_correction(np.eye(3))

    for channel in range(3):
        assert corrected_values(channel) == pytest.approx(SUM_COL_VALS)


def test_the_corrected_values_keep_their_fraction(tmp_path, monkeypatch):
    write_binaries(tmp_path, monkeypatch, values=[1000.0])

    apply_color_correction(np.diag([3.0, 3.0, 3.0]))

    assert corrected_values(0) == pytest.approx([1000.0 / 3])
