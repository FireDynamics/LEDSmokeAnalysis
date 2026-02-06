from __future__ import annotations

import os
import re
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import fdsreader as fds

from ledsa.core.ConfigData import ConfigData
from ledsa.analysis.Experiment import Experiment
from ledsa.analysis.ExperimentData import ExperimentData

if TYPE_CHECKING:
    import pandas as pd


class Ledsa2fds:
    """
    Build FDS &DEVC lines (PATH OBSCURATION) from one or more LEDSA experiment datasets.

    The resulting device IDs follow the pattern:
    LEDSA_{name}_line_{line_id}_led_{led_id}

    """


    def __init__(self) -> None:
        """
        :ivar all_experiments: Mapping from dataset key (e.g. "{name}_{array_idx}") to Experiment instance.
        :vartype all_experiments: Dict[str, Experiment]
        """
        self.all_experiments: Dict[str, Experiment] = {}

    def add_experiment_dataset(self, path: str, name: str) -> None:
        """
        Load a LEDSA dataset and create an Experiment instance for every LED array.

        :param path: Directory containing the dataset/config needed by ConfigData and ExperimentData.
        :type path: str
        :param name: Name of the LEDSA experiment dataset (usually camera name)
        :type name: str
        """
        old_cwd = os.getcwd()
        try:
            os.chdir(path)

            config = ConfigData()
            num_arrays = int(config['analyse_positions']['num_arrays'])

            exp_data = ExperimentData()
            exp_data.request_config_parameters()
            camera = exp_data.camera

            for led_array in range(num_arrays):
                # Store experiments by a unique key (dataset name + led array index)
                extended_name = f"{name}_{led_array}"

                # layers=None because we only need camera + LED geometry here
                exp = Experiment(layers=None, led_array=led_array, camera=camera, merge_led_arrays='None')
                self.all_experiments[extended_name] = exp

        finally:
            os.chdir(old_cwd)


    def create_transmission_devices_file(self, path: str) -> None:
        """
        Create a text file containing FDS &DEVC lines for all camera->LED transmission paths.

        :param path: Output directory for the device file.
        :type path: str
        """
        file_path = os.path.join(path, 'LEDSA_transmission_devices.txt')
        output_file = open(file_path, "w")
        for extended_name, exp in self.all_experiments.items():
            name = extended_name.rsplit('_', 1)[0]
            leds = exp.leds
            camera = exp.camera
            for led in leds:
                led_array = exp.led_array
                device_string = _get_device_str(name, led_array, camera, led)
                output_file.write(device_string + "\n")
        output_file.close()


def create_led_positions_file(img_id: int, transmission_data: "pd.Series", path: str, name: str) -> None:
    """
    Create a LEDSA-style led_positions CSV for one "image" from a single row of FDS device output.
    The FDS-related LEDSA data is treated as channel 3.

    This writes:
        analysis/channel3/{img_id}_led_positions.csv

    :param img_id: Identifier for the (virtual) image, here is just a rolling index.
    :type img_id: int
    :param transmission_data: A single row (Series) containing a 'Time' field and device columns.
    :type transmission_data: pandas.Series
    :param path: Output root directory.
    :type path: str
    :param name: Dataset name used in DEVC IDs (used to filter matching device columns).
    :type name: str
    """
    file_path = os.path.join(path, 'analysis', 'channel3', f'{img_id}_led_positions.csv')
    out_file = open(file_path, 'w')

    header = _create_header(img_id=img_id, time=transmission_data['Time'], basename=path)
    out_file.write(header)
    sub = transmission_data.filter(regex=f'LEDSA_{name}.*')
    for col_name, led_transmission in sub.items():
        line_id, led_id = _get_device_params_from_str(col_name)
        transmission_string = str(led_id) + "," + str(line_id) + "," + str(100-led_transmission) + ",0,0" + '\n'
        out_file.write(transmission_string)
    out_file.close()


class Fds2ledsa:
    """
    Convert FDS device output (DEVC) into LEDSA-style per-image CSV files.
    """

    def __init__(self) -> None:
        """
        :ivar devc_data: Device data table (time series) loaded via fdsreader.
        :vartype devc_data
        """
        self.devc_data = None

    def load_devc_data(self, path: str) -> None:
        """
        Load FDS simulation device output using fdsreader.

        :param path: Path (directory) that fdsreader.Simulation expects.
        :type path: str
        """
        sim = fds.Simulation(path)
        self.devc_data = sim.devices.to_pandas_dataframe()

    def create_ledsa_dataset(self, path: str, name: str) -> None:
        """
        Create LEDSA-style led_positions CSV files for every time step in devc_data.

        :param path: Output root directory (will create analysis/channel3 below).
        :type path: str
        :param name: Dataset name used in DEVC IDs (used to filter matching device columns).
        :type name: str
        :raises RuntimeError: If no device data has been loaded.
        """
        if self.devc_data is None:
            raise RuntimeError("No device data loaded. Call load_devc_data(path) first.")

        os.makedirs(os.path.join(path, 'analysis', 'channel3'), exist_ok=True)

        # Each row is treated as one "image"/time step; we create one led_positions file per row
        for index, row in self.devc_data.iterrows():
            create_led_positions_file(index + 1, row, path, name)


def _create_header(img_id: int, time: float, basename: str) -> str:
    """
    Create a header for the analysis result file.

    :param img_id: Identifier for the image.
    :type img_id: int
    :param time: Simulation time in seconds.
    :type time: float
    :param basename: Root path (written into header for traceability).
    :type basename: str
    :return: Header string for the LEDSA result file.
    :rtype: str
    """
    out_str = f'# image root = {basename}, virtual image id = {img_id},'
    out_str += f"channel = 3, "
    out_str += f"time[s] = {time:.1f}\n"
    out_str += "# id,line,sum_col_value,average_col_value,max_col_value\n"
    return out_str


def _get_device_str(name: str, line_id: int, camera, led) -> str:
    """
    Create an FDS &DEVC definition for PATH OBSCURATION between camera and LED.

    :param name: Dataset name included in the device ID.
    :type name: str
    :param line_id: LED array / line identifier.
    :type line_id: int
    :param camera: Camera object with pos_x/pos_y/pos_z attributes.
    :param led: LED object with id, pos_x/pos_y/pos_z attributes.
    :return: A single &DEVC line as string.
    :rtype: str
    """
    return (
        f"&DEVC XB={camera.pos_x:.2f},{led.pos_x:.2f},"
        f"{camera.pos_y:.2f},{led.pos_y:.2f},"
        f"{camera.pos_z:.2f},{led.pos_z:.2f}, "
        f"QUANTITY='PATH OBSCURATION', ID='LEDSA_{name}_line_{line_id}_led_{led.id}'/"
    )


def _get_device_params_from_str(devc_str: str) -> Tuple[int, int]:
    """
    Parse (line_id, led_id) from a device ID string.

    Accepts "ID='LEDSA_<name>_line_<line_id>_led_<led_id>'"

    :param devc_str: Device string to parse.
    :type devc_str: str
    :return: Tuple of (line_id, led_id).
    :rtype: Tuple[int, int]
    :raises ValueError: If the expected pattern is not found.
    """
    prefix = "ID='LEDSA_"
    suffix = "'"

    core = devc_str.removeprefix(prefix).removesuffix(suffix)

    _, sep, rest = core.partition("_line_")
    if not sep:
        raise ValueError(f"Missing '_line_': {devc_str!r}")

    line_id, sep, led_part = rest.partition("_led_")
    if not sep:
        raise ValueError(f"Missing '_led_': {devc_str!r}")

    led_id = led_part
    return int(line_id), int(led_id)


