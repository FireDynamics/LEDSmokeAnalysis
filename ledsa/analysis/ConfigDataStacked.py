import configparser as cp
from pathlib import Path
from typing import Dict, List, Optional

from ledsa.analysis.ConfigDataAnalysis import DEFAULT_REFERENCE_PROPERTY, parse_reference_properties
from ledsa.analysis.temperature_correction import TEMPERATURE_CORRECTION_CHANNELS
from ledsa.core.config_file import save_config


class ConfigDataStacked(cp.ConfigParser):
    """
    Configuration for stacked (multi-camera) extinction coefficient computation.

    Reads ``config_stacked.ini``, which defines:

    - Global solver settings and domain parameters (``[DEFAULT]``)
    - One ``[simulation_N]`` section per camera/LEDSA run
    - A ``[led_arrays]`` section that maps virtual LED-array labels to
      per-simulation LED-array IDs

    Example file layout::

        [DEFAULT]
        solver            = linear
        lambda_reg        = 1e-3
        camera_channels   = 0 1 2
        num_layers        = 20
        domain_bounds     = 0.0 3.0
        reference_property = sum_col_val bgsub_sum_col_val
        num_ref_images    = 10
        ref_img_indices   = None
        average_images    = False
        n_summarize       = 2
        num_cores         = 1
        output_path       = .
        time_sync_tolerance = 0.5

        [simulation_0]
        path             = /data/exp01/cam0
        camera_position  = 1.0 0.0 1.5

        [simulation_1]
        path             = /data/exp01/cam1
        camera_position  = -1.0 0.0 1.5
        # num_ref_images and ref_img_indices may be set per camera and override [DEFAULT]
        ref_img_indices  = 385 386 387

        [led_arrays]
        # integer_id = sim_index:led_array_id  [sim_index:led_array_id ...]
        # sim_index refers to [simulation_N]; led_array_id is the integer ID from that run.
        # List all sim:id pairs that observe the same physical array to stack their rays.
        0 = 0:2 1:3
        1 = 0:4 1:5

    """

    DEFAULT_FILENAME = 'config_stacked.ini'

    def __init__(self, filename: Optional[str] = None, load_config_file: bool = True):
        """
        :param filename: Path to the INI file. Defaults to ``config_stacked.ini``
            in the current working directory.
        :type filename: str or None
        :param load_config_file: Whether to load the file on construction.
        :type load_config_file: bool
        """
        cp.ConfigParser.__init__(self, allow_no_value=True)
        self._filename = filename or self.DEFAULT_FILENAME
        if load_config_file:
            self.load()

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the INI file from disk.

        :raises SystemExit: if the file is not found.
        """
        try:
            with open(self._filename, encoding='utf-8-sig') as config_file:
                self.read_file(config_file)
        except FileNotFoundError:
            print(
                f'{self._filename} not found in the working directory! '
                'Please create it according to the documentation.'
            )
            exit(1)
        print(f'{self._filename} loaded.')

    def save(self) -> None:
        """Write the current configuration back to disk, keeping the comments of the file."""
        save_config(self, self._filename)
        print(f'{self._filename} saved.')

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def get_list_of_values(
        self,
        section: str,
        option: str,
        dtype=int,
        fallback=None,
    ):
        """Return a typed list for *section*/*option*, or *fallback*.

        The value ``None`` (case-insensitive) or an empty string yields
        ``None`` rather than a list.
        """
        if not self.has_option(section, option):
            return fallback
        raw = self.get(section, option, fallback=None)
        if raw is None:
            return fallback
        raw = raw.strip()
        if raw.lower() == 'none' or raw == '':
            return None
        try:
            return [dtype(item) for item in raw.split()]
        except Exception as exc:
            print(f'Config parse error [{section}].{option}={raw!r}: {exc} — using fallback={fallback!r}')
            return fallback

    # ------------------------------------------------------------------
    # Simulation sections
    # ------------------------------------------------------------------

    def get_simulation_sections(self) -> List[str]:
        """Return all section names that start with ``simulation_``."""
        return sorted(s for s in self.sections() if s.startswith('simulation_'))

    def get_num_simulations(self) -> int:
        """Number of simulation sections defined in the config."""
        return len(self.get_simulation_sections())

    def get_simulation_path(self, sim_idx: int) -> Path:
        """Filesystem path for simulation *sim_idx*."""
        return Path(self.get(f'simulation_{sim_idx}', 'path'))

    def get_camera_position(self, sim_idx: int) -> List[float]:
        """Camera position ``[x, y, z]`` for simulation *sim_idx*."""
        return self.get_list_of_values(f'simulation_{sim_idx}', 'camera_position', dtype=float)

    def get_ref_img_indices(self, sim_idx: int) -> Optional[List[int]]:
        """Reference image IDs of simulation *sim_idx*; ``None`` means use
        :meth:`get_num_ref_images`.

        Image IDs are counted per camera and the cameras run at their own cadence, so the
        images taken before the fire differ from camera to camera. A value in
        ``[simulation_N]`` therefore overrides the one in ``[DEFAULT]`` for that camera,
        a missing value is inherited from ``[DEFAULT]``, and ``None`` in a section switches
        that camera back to :meth:`get_num_ref_images` even if ``[DEFAULT]`` holds a list.
        """
        return self.get_list_of_values(f'simulation_{sim_idx}', 'ref_img_indices',
                                       dtype=int, fallback=None)

    def get_num_ref_images(self, sim_idx: int) -> int:
        """Number of reference images of simulation *sim_idx*, inherited from ``[DEFAULT]``."""
        return int(self.get(f'simulation_{sim_idx}', 'num_ref_images', fallback='10'))

    # ------------------------------------------------------------------
    # LED-array mapping
    # ------------------------------------------------------------------

    def get_led_array_ids(self) -> List[int]:
        """Return the virtual LED-array IDs defined in ``[led_arrays]`` as integers.

        ``ConfigParser.options()`` merges DEFAULT keys into every section, so
        we explicitly subtract the DEFAULT keys to get only the keys that are
        actually written inside the ``[led_arrays]`` block.
        """
        default_keys = set(k.lower() for k in self.defaults())
        return sorted(
            int(k) for k in self.options('led_arrays') if k not in default_keys
        )

    def get_led_array_mapping(self, led_array_id: int) -> Dict[int, int]:
        """
        For a virtual array *led_array_id* return ``{sim_idx: led_array_id}``.

        Config value format: ``"0:2 1:3"``
        """
        raw = self.get('led_arrays', str(led_array_id)).strip()
        mapping: Dict[int, int] = {}
        for token in raw.split():
            sim_str, arr_str = token.split(':')
            mapping[int(sim_str)] = int(arr_str)
        return mapping

    # ------------------------------------------------------------------
    # Typed property accessors for global parameters
    # ------------------------------------------------------------------

    @property
    def solver(self) -> str:
        """Inversion method: ``'linear'`` or ``'nonlinear'``."""
        return self.get('DEFAULT', 'solver', fallback='linear')

    @property
    def lambda_reg(self) -> float:
        """Tikhonov regularisation parameter (linear solver)."""
        return float(self.get('DEFAULT', 'lambda_reg', fallback='1e-3'))

    @property
    def weighting_preference(self) -> float:
        """Preference weighting for the nonlinear solver."""
        return float(self.get('DEFAULT', 'weighting_preference', fallback='-6e-3'))

    @property
    def weighting_curvature(self) -> float:
        """Curvature weighting for the nonlinear solver."""
        return float(self.get('DEFAULT', 'weighting_curvature', fallback='1e-2'))

    @property
    def num_iterations(self) -> int:
        """Maximum iterations for the nonlinear solver."""
        return int(self.get('DEFAULT', 'num_iterations', fallback='1000'))

    @property
    def reference_properties(self) -> List[str]:
        """Column names used as the LED intensity measure, one stacked analysis each."""
        return parse_reference_properties(
            self.get('DEFAULT', 'reference_property', fallback=DEFAULT_REFERENCE_PROPERTY))

    @property
    def reference_property(self) -> str:
        """First of :attr:`reference_properties`, the quantity of a single analysis."""
        return self.reference_properties[0]

    @property
    def num_ref_images(self) -> int:
        """Number of reference images used to compute I₀, as set in ``[DEFAULT]``.

        Use :meth:`get_num_ref_images` to get the value of a single camera.
        """
        return int(self.get('DEFAULT', 'num_ref_images', fallback='10'))

    @property
    def ref_img_indices(self) -> Optional[List[int]]:
        """Explicit reference image indices of ``[DEFAULT]``; ``None`` → use *num_ref_images*.

        Use :meth:`get_ref_img_indices` to get the indices of a single camera.
        """
        return self.get_list_of_values('DEFAULT', 'ref_img_indices', dtype=int, fallback=None)

    @property
    def average_images(self) -> bool:
        """Whether to average the intensities over :attr:`n_summarize` consecutive images."""
        return self.getboolean('DEFAULT', 'average_images', fallback=False)

    @property
    def n_summarize(self) -> int:
        """Number of consecutive images summarized into one when :attr:`average_images` is set.

        The reference images are kept as they are, so the first :attr:`num_ref_images` images of
        every camera are not summarized.
        """
        return int(self.get('DEFAULT', 'n_summarize', fallback='2'))

    @property
    def num_cores(self) -> int:
        """Number of CPU cores for parallel processing."""
        return int(self.get('DEFAULT', 'num_cores', fallback='1'))

    @property
    def num_layers(self) -> int:
        """Number of horizontal smoke layers."""
        return int(self.get('DEFAULT', 'num_layers', fallback='20'))

    @property
    def domain_bounds(self) -> List[float]:
        """``[z_min, z_max]`` of the computational domain in metres."""
        return self.get_list_of_values('DEFAULT', 'domain_bounds', dtype=float)

    @property
    def output_path(self) -> Path:
        """Root path for all stacked-analysis output files."""
        return Path(self.get('DEFAULT', 'output_path', fallback='.'))

    @property
    def camera_channels(self) -> List[int]:
        """List of colour channels (0 = R, 1 = G, 2 = B) to process."""
        return self.get_list_of_values('DEFAULT', 'camera_channels', dtype=int, fallback=[0])

    @property
    def time_sync_tolerance(self) -> float:
        """Maximum allowed difference [s] when matching images across simulations."""
        return float(self.get('DEFAULT', 'time_sync_tolerance', fallback='0.5'))

    def get_reduction_file(self, channel: int) -> Optional[str]:
        """Path of the reduction factor file of the temperature correction of *channel*.

        The factors apply to the LEDs themselves, so all cameras of a stacked inversion share
        the file of a channel. A config file written before the temperature correction existed
        has no ``[temperature_correction]`` section, which asks for no correction just like the
        option left at ``None``.

        :param channel: The camera channel.
        :return: The configured path, ``None`` if the channel is analysed without the correction.
        """
        option = f'reduction_file_channel_{channel}'
        if not self.has_section('temperature_correction'):
            return None
        if not self.has_option('temperature_correction', option):
            return None
        raw = self.get('temperature_correction', option, fallback='None').strip()
        if raw.lower() == 'none' or raw == '':
            return None
        return raw

    # ------------------------------------------------------------------
    # Factory: write a template config to disk
    # ------------------------------------------------------------------

    @classmethod
    def create_template(cls, filename: str = DEFAULT_FILENAME, num_simulations: int = 2) -> None:
        """Write an annotated template ``config_stacked.ini`` to disk.

        :param filename: Destination file path.
        :param num_simulations: Number of simulation blocks to include.
        """
        cfg = cp.ConfigParser(allow_no_value=True)

        cfg['DEFAULT'] = {}
        cfg.set('DEFAULT', '# ── Solver ────────────────────────────────────────────────────────')
        cfg.set('DEFAULT', "# 'linear' (Tikhonov-NNLS) or 'nonlinear' (TNC minimisation)")
        cfg['DEFAULT']['solver'] = 'linear'
        cfg.set('DEFAULT', '# Regularisation for linear solver')
        cfg['DEFAULT']['lambda_reg'] = '1e-3'
        cfg.set('DEFAULT', '# Nonlinear solver weights (ignored when solver = linear)')
        cfg['DEFAULT']['weighting_preference'] = '-6e-3'
        cfg['DEFAULT']['weighting_curvature'] = '1e-2'
        cfg['DEFAULT']['num_iterations'] = '1000'
        cfg.set('DEFAULT', '# ── Image / reference ──────────────────────────────────────────────')
        cfg.set('DEFAULT', '# Quantities the extinction coefficients are computed from, separated by spaces')
        cfg['DEFAULT']['reference_property'] = 'sum_col_val'
        cfg['DEFAULT']['num_ref_images'] = '10'
        cfg.set('DEFAULT', "# Explicit reference image indices (overrides num_ref_images); 'None' to disable")
        cfg['DEFAULT']['ref_img_indices'] = 'None'
        cfg['DEFAULT']['average_images'] = 'False'
        cfg.set('DEFAULT', '# Consecutive images summarized into one when average_images is True')
        cfg['DEFAULT']['n_summarize'] = '2'
        cfg.set('DEFAULT', '# ── Channels / parallelism ─────────────────────────────────────────')
        cfg.set('DEFAULT', '# Space-separated list: 0 = R, 1 = G, 2 = B')
        cfg['DEFAULT']['camera_channels'] = '0'
        cfg['DEFAULT']['num_cores'] = '1'
        cfg.set('DEFAULT', '# ── Spatial domain ─────────────────────────────────────────────────')
        cfg['DEFAULT']['num_layers'] = '20'
        cfg.set('DEFAULT', '# Lower and upper z-bounds [m] of the computational domain')
        cfg['DEFAULT']['domain_bounds'] = '0.0 3.0'
        cfg.set('DEFAULT', '# ── Output ──────────────────────────────────────────────────────────')
        cfg.set('DEFAULT', "# Directory for stacked-analysis results; '.' = current working directory")
        cfg['DEFAULT']['output_path'] = '.'
        cfg.set('DEFAULT', '# Maximum time difference [s] for matching images across cameras')
        cfg['DEFAULT']['time_sync_tolerance'] = '0.5'

        for i in range(num_simulations):
            section = f'simulation_{i}'
            cfg[section] = {}
            cfg.set(section, f'# Path to the LEDSA run directory for camera {i}')
            cfg[section]['path'] = f'/path/to/sim{i}'
            cfg.set(section, '# Global X Y Z position [m] of camera')
            cfg[section]['camera_position'] = '0.0 0.0 1.5'
            cfg.set(section, '# Reference images of this camera; both options override the ones above')
            cfg.set(section, '# Image IDs are counted per camera, so choose images from the same time')
            cfg.set(section, '# window before the fire in every camera')
            cfg.set(section, '# ref_img_indices = 385 386 387   (None falls back to num_ref_images)')
            cfg.set(section, '# num_ref_images = 10')

        cfg['temperature_correction'] = {}
        cfg.set('temperature_correction', '# Correction of the LED intensities for the temperature of the LEDs.')
        cfg.set('temperature_correction', '# Path of the reduction factor file of every camera channel. A channel with a file')
        cfg.set('temperature_correction', '# is corrected with it, a channel set to None is analysed as extracted. The factors')
        cfg.set('temperature_correction', '# are computed outside LEDSA and read from a CSV file holding the experiment time [s]')
        cfg.set('temperature_correction', '# in its first column, the LED height [m] in its header and the factor the intensity')
        cfg.set('temperature_correction', '# of that time and height is reduced by in its cells. They apply to the LEDs, so all')
        cfg.set('temperature_correction', '# cameras of the stacked inversion share the file of a channel.')
        for channel in TEMPERATURE_CORRECTION_CHANNELS:
            cfg['temperature_correction'][f'reduction_file_channel_{channel}'] = 'None'

        cfg['led_arrays'] = {}
        cfg.set('led_arrays', '# Each line maps a virtual array ID (integer) to one or more  sim_index:led_array_id  tokens.')
        cfg.set('led_arrays', '# The integer key is the ID used in output filenames and postprocessing.')
        cfg.set('led_arrays', '# sim_index refers to the [simulation_N] block; led_array_id is the integer ID')
        cfg.set('led_arrays', '# assigned by that simulation\'s LEDSA run (see analysis/led_array_indices_<id>.csv).')
        cfg.set('led_arrays', '# List all sim:id pairs that observe the same physical array to stack their rays.')
        cfg.set('led_arrays', '# Arrays seen by only one camera need just one token.')
        cfg['led_arrays']['0'] = '0:0 1:0'
        cfg['led_arrays']['1'] = '0:1 1:1'

        with open(filename, 'w', encoding='utf-8') as f:
            cfg.write(f)
        print(f'Template written to {filename}')