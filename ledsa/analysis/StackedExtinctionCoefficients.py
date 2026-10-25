"""
Stacked extinction coefficient computation across multiple cameras/LEDSA runs.

Usage example::

    from ledsa.analysis.ConfigDataStacked import ConfigDataStacked
    from ledsa.analysis.StackedExtinctionCoefficients import StackedExtinctionCoefficients

    cfg = ConfigDataStacked()                    # reads config_stacked.ini

    for arr_id in cfg.get_led_array_ids():
        for channel in cfg.camera_channels:
            sec = StackedExtinctionCoefficients(cfg, arr_id, channel)
            sec.calc_and_set_coefficients()
            sec.save()
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ledsa.analysis.ConfigDataStacked import ConfigDataStacked
from ledsa.analysis.Experiment import Camera, Experiment, Layers
from ledsa.analysis.ExtinctionCoefficientsLinear import ExtinctionCoefficientsLinear
from ledsa.analysis.ExtinctionCoefficientsNonLinear import ExtinctionCoefficientsNonLinear
from importlib.metadata import version


class StackedExtinctionCoefficients:
    """
    Joint inversion of extinction coefficients using ray data stacked across
    multiple cameras (LEDSA simulation directories).

    For a given virtual LED-array ID and colour channel the class:

    1. Builds one :class:`~ledsa.analysis.Experiment.Experiment` per
       (simulation, ``led_array_id``) pair.
    2. Constructs a stacked system matrix by vertically concatenating the
       per-experiment distance-per-layer matrices and writes it once per
       virtual array (channel-independent) to ``<output_path>/``.
    3. Computes the common time axis across all simulations, writes a
       time-alignment report to ``<output_path>/``, and stores it for reuse.
    4. Solves the augmented Beer–Lambert system with the configured solver.
    5. Writes extinction coefficient CSVs to
       ``<output_path>/analysis/extinction_coefficients/<solver>/``.

    :ivar config: Stacked-analysis configuration.
    :vartype config: ConfigDataStacked
    :ivar led_array_id: Virtual LED-array ID (integer key in ``[led_arrays]``).
    :vartype led_array_id: int
    :ivar channel: Colour channel index.
    :vartype channel: int
    :ivar layers: Shared layer discretisation.
    :vartype layers: Layers
    :ivar led_array_mapping: ``{sim_idx: led_array_id}`` for this virtual array.
    :vartype led_array_mapping: dict[int, int]
    :ivar sub_experiments: Per-simulation :class:`Experiment` objects.
    :vartype sub_experiments: list[Experiment]
    :ivar distances_per_led_and_layer: Stacked distance matrix (n_leds_total × n_layers).
    :vartype distances_per_led_and_layer: np.ndarray
    :ivar ref_intensities: Stacked reference intensities (n_leds_total,).
    :vartype ref_intensities: np.ndarray
    :ivar coefficients_per_image_and_layer: Computed extinction coefficients.
    :vartype coefficients_per_image_and_layer: list[np.ndarray]
    :ivar common_times: Experiment times [s] shared by all simulations.
    :vartype common_times: np.ndarray or None
    """

    def __init__(self, config: ConfigDataStacked, led_array_id: int, channel: int) -> None:
        """
        :param config: Parsed ``config_stacked.ini``.
        :type config: ConfigDataStacked
        :param led_array_id: Integer key defined in the ``[led_arrays]`` section.
        :type led_array_id: int
        :param channel: Colour channel (0 = R, 1 = G, 2 = B).
        :type channel: int
        """
        self.config = config
        self.led_array_id = led_array_id
        self.channel = channel

        self.layers = Layers(config.num_layers, *config.domain_bounds)
        self.led_array_mapping: Dict[int, int] = config.get_led_array_mapping(led_array_id)

        self.sub_experiments: List[Experiment] = []
        for sim_idx, arr_id in sorted(self.led_array_mapping.items()):
            cam_pos = config.get_camera_position(sim_idx)
            exp = Experiment(
                layers=self.layers,
                led_array=arr_id,
                camera=Camera(*cam_pos),
                path=config.get_simulation_path(sim_idx),
                channel=channel,
                merge_led_arrays='None',
            )
            self.sub_experiments.append(exp)

        self._sub_solvers: List = [
            self._build_sub_solver(exp) for exp in self.sub_experiments
        ]

        self.distances_per_led_and_layer: np.ndarray = np.array([])
        self.ref_intensities: np.ndarray = np.array([])
        self.coefficients_per_image_and_layer: List[np.ndarray] = []
        self.common_times: Optional[np.ndarray] = None
        self._time_maps: Optional[List[Dict[int, float]]] = None  # set by _compute_time_alignment
        self.solver: str = config.solver

    # ------------------------------------------------------------------
    # Sub-solver factory
    # ------------------------------------------------------------------

    def _build_sub_solver(self, experiment: Experiment):
        """Create a single-experiment solver instance for *experiment*."""
        cfg = self.config
        if cfg.solver == 'linear':
            return ExtinctionCoefficientsLinear(
                experiment=experiment,
                reference_property=cfg.reference_property,
                num_ref_imgs=cfg.num_ref_images,
                ref_img_indices=cfg.ref_img_indices,
                average_images=cfg.average_images,
                lambda_reg=cfg.lambda_reg,
            )
        elif cfg.solver == 'nonlinear':
            return ExtinctionCoefficientsNonLinear(
                experiment=experiment,
                reference_property=cfg.reference_property,
                num_ref_imgs=cfg.num_ref_images,
                ref_img_indices=cfg.ref_img_indices,
                average_images=cfg.average_images,
                weighting_preference=cfg.weighting_preference,
                weighting_curvature=cfg.weighting_curvature,
                num_iterations=cfg.num_iterations,
            )
        else:
            raise ValueError(
                f'Unknown solver {cfg.solver!r}. Choose "linear" or "nonlinear".'
            )

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def calc_and_set_coefficients(self) -> None:
        """
        Compute extinction coefficients for all time-aligned images.

        Populates :attr:`coefficients_per_image_and_layer` and
        :attr:`common_times`.
        """
        self._ensure_stacked_variables()

        ref_prop = self.config.reference_property
        for _, aligned_intensities in self._iter_aligned_images(ref_prop):
            rel_intensities = aligned_intensities / self.ref_intensities
            if self.config.solver == 'nonlinear' and self.coefficients_per_image_and_layer:
                self._sub_solvers[0].coefficients_per_image_and_layer = (
                    self.coefficients_per_image_and_layer
                )
            sigmas = self._sub_solvers[0].calc_coefficients_of_img(rel_intensities)
            self.coefficients_per_image_and_layer.append(sigmas)

    def calc_and_set_coefficients_mp(self, cores: Optional[int] = None) -> None:
        """Parallel computation using multiprocessing.

        :param cores: Number of worker processes. Defaults to ``config.num_cores``.
        :type cores: int or None
        """
        from multiprocessing import Pool

        if cores is None:
            cores = self.config.num_cores

        self._ensure_stacked_variables()
        ref_prop = self.config.reference_property
        all_rel_intensities = [
            aligned / self.ref_intensities
            for _, aligned in self._iter_aligned_images(ref_prop)
        ]

        with Pool(processes=cores) as pool:
            self.coefficients_per_image_and_layer = pool.map(
                self._sub_solvers[0].calc_coefficients_of_img, all_rel_intensities
            )

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_stacked_variables(self) -> None:
        """Load/compute distance matrix, time alignment, and reference intensities."""
        if self.distances_per_led_and_layer.size == 0:
            self._build_stacked_distance_matrix()
        if self._time_maps is None:
            self._compute_time_alignment()
        if self.ref_intensities.size == 0:
            self._build_stacked_ref_intensities()

    def _build_stacked_distance_matrix(self) -> None:
        """
        Populate each sub-solver's distance matrix, then stack them.

        The stacked distance matrix is written once per virtual array
        (channel-independent) to::

            <output_path>/led_array_<id>_distances_per_layer.txt

        A guard prevents redundant writes when multiple channels are processed.
        """
        for sim_idx, (sub, arr_id) in enumerate(
            zip(self._sub_solvers,
                [self.led_array_mapping[i] for i in sorted(self.led_array_mapping)])
        ):
            exp = sub.experiment
            if exp.num_leds is None:
                raise FileNotFoundError(
                    f"Could not load LED data for simulation {sim_idx} "
                    f"(led_array_id={arr_id}, path={exp.path}).\n"
                    f"Expected files:\n"
                    f"  {exp.path / 'analysis' / f'led_array_indices_{arr_id:03d}.csv'}\n"
                    f"  {exp.path / 'analysis' / 'led_search_areas_with_coordinates.csv'}\n"
                    "Make sure LEDSA steps 1-3 have been completed in that directory "
                    "and that the led_array_id in [led_arrays] is correct."
                )
            sub.set_all_member_variables(save_distances=False)

        self.distances_per_led_and_layer = np.vstack(
            [sub.distances_per_led_and_layer for sub in self._sub_solvers]
        )

        dist_file = self.config.output_path / f'led_array_{self.led_array_id}_distances_per_layer.txt'
        if not dist_file.exists():
            self.config.output_path.mkdir(parents=True, exist_ok=True)
            np.savetxt(dist_file, self.distances_per_led_and_layer)

        self._sub_solvers[0].distances_per_led_and_layer = self.distances_per_led_and_layer

    def _build_stacked_ref_intensities(self) -> None:
        """Concatenate per-sub-solver reference intensities."""
        self.ref_intensities = np.concatenate(
            [sub.ref_intensities for sub in self._sub_solvers]
        )

    def _compute_time_alignment(self) -> None:
        """
        Compute the common time axis across all simulations and write a
        time-alignment report to::

            <output_path>/time_alignment_led_array_<id>.txt

        The report is written once per virtual array (channel-independent).
        Sets :attr:`_time_maps` and :attr:`common_times`.
        """
        tol = self.config.time_sync_tolerance

        self._time_maps = [
            self._load_time_map(sub.experiment, sub.average_images)
            for sub in self._sub_solvers
        ]

        all_time_arrays = [np.array(list(tm.values())) for tm in self._time_maps]
        common_times = self._intersect_time_axes(all_time_arrays, tol)

        if len(common_times) == 0:
            raise RuntimeError(
                'No overlapping experiment times found across simulations for '
                f'led_array_id={self.led_array_id}. '
                'Check time_sync_tolerance in config_stacked.ini.'
            )
        self.common_times = common_times

        print(
            f'  [array {self.led_array_id}] '
            f'{len(common_times)} time-aligned images from '
            f'{self.config.get_num_simulations()} simulations.'
        )

        report_file = self.config.output_path / f'time_alignment_led_array_{self.led_array_id}.txt'
        if not report_file.exists():
            self._write_time_alignment_report(report_file, self._time_maps, all_time_arrays, common_times, tol)

    def _write_time_alignment_report(
        self,
        path: Path,
        time_maps: List[Dict[int, float]],
        all_time_arrays: List[np.ndarray],
        common_times: np.ndarray,
        tol: float,
    ) -> None:
        """
        Write a plain-text time-alignment report.

        Includes a per-image table showing, for every common time step, which
        image ID was selected from each simulation and the corresponding time
        difference.

        :param path: Destination file path.
        :param time_maps: ``{img_id: time}`` per simulation.
        :param all_time_arrays: Flat time arrays per simulation.
        :param common_times: Intersected common time axis.
        :param tol: Matching tolerance used [s].
        """
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        sim_indices = sorted(self.led_array_mapping)

        lines = [
            f'Time-alignment report — led_array_id {self.led_array_id}',
            f'Generated : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            f'Tolerance : {tol} s',
            '',
            'Simulations',
            '-----------',
        ]

        for sim_idx, arr_id, times in zip(
            sim_indices, self.led_array_mapping.values(), all_time_arrays
        ):
            sim_path = self.config.get_simulation_path(sim_idx)
            t_min = float(times.min()) if len(times) else float('nan')
            t_max = float(times.max()) if len(times) else float('nan')
            lines += [
                f'  sim {sim_idx}  (led_array_id={arr_id})',
                f'    path   : {sim_path}',
                f'    images : {len(times)}',
                f'    range  : {t_min:.1f} s – {t_max:.1f} s',
            ]

        lines += [
            '',
            'Alignment result',
            '----------------',
            f'  Common images : {len(common_times)}',
        ]
        if len(common_times):
            lines += [
                f'  Time range    : {float(common_times[0]):.1f} s – {float(common_times[-1]):.1f} s',
            ]

        lines += ['', 'Per-simulation match rate', '-------------------------']
        for sim_idx, times in zip(sim_indices, all_time_arrays):
            matched = sum(1 for t in times if np.any(np.abs(common_times - t) <= tol))
            pct = 100.0 * matched / len(times) if len(times) else 0.0
            lines.append(f'  sim {sim_idx} : {matched}/{len(times)} images matched ({pct:.1f} %)')

        # ------------------------------------------------------------------
        # Per-image matching table
        # ------------------------------------------------------------------
        # Build reverse map: time → img_id for each simulation
        rev_maps: List[Dict[float, int]] = [
            {t: img_id for img_id, t in tm.items()} for tm in time_maps
        ]

        # Column widths
        col_t_common = 14          # "common t / s"
        col_per_sim  = 26          # "img_id  t_sim / s   Δt / s" per simulation

        # Header
        sim_headers = ''.join(
            f'  {"sim " + str(si):^{col_per_sim}}'
            for si in sim_indices
        )
        sub_headers = ''.join(
            f'  {"img_id":>8}  {"t_sim / s":>10}  {"Δt / s":>6}'
            for _ in sim_indices
        )
        separator = '-' * (col_t_common + len(sim_indices) * (col_per_sim + 2))

        lines += [
            '',
            'Per-image match table',
            '---------------------',
            f'  {"common t / s":>{col_t_common}}' + sim_headers,
            f'  {" " * col_t_common}' + sub_headers,
            '  ' + separator,
        ]

        for t_common in common_times:
            row = f'  {t_common:>{col_t_common}.2f}'
            for tm in time_maps:
                best_id = self._nearest_img_id(tm, t_common, tol)
                if best_id is not None:
                    t_sim = tm[best_id]
                    dt    = t_sim - t_common
                    row += f'  {best_id:>8d}  {t_sim:>10.2f}  {dt:>+6.2f}'
                else:
                    row += f'  {"—":>8}  {"—":>10}  {"—":>6}'
            lines.append(row)

        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(f'  Time-alignment report: {path}')

    # ------------------------------------------------------------------
    # Time-aligned image iteration
    # ------------------------------------------------------------------

    def _iter_aligned_images(self, reference_property: str):
        """
        Yield ``(virtual_img_id, stacked_intensity_vector)`` for every
        time-aligned image set, using the pre-computed :attr:`common_times`
        and :attr:`_time_maps`.
        """
        tol = self.config.time_sync_tolerance

        for v_img_id, t_common in enumerate(self.common_times, start=1):
            blocks: List[np.ndarray] = []
            skip = False
            for sub, time_map in zip(self._sub_solvers, self._time_maps):
                orig_id = self._nearest_img_id(time_map, t_common, tol)
                if orig_id is None:
                    print(
                        f'  Warning: no image within {tol} s of t={t_common:.2f} s '
                        f'in simulation at {sub.experiment.path} — skipping time step.'
                    )
                    skip = True
                    break
                try:
                    img_slice = sub.calculated_img_data.loc[orig_id]
                except KeyError:
                    print(
                        f'  Warning: img_id={orig_id} missing in calculated_img_data '
                        f'for {sub.experiment.path} — skipping time step.'
                    )
                    skip = True
                    break
                blocks.append(img_slice[reference_property].to_numpy())

            if skip:
                continue
            yield v_img_id, np.concatenate(blocks)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self) -> None:
        """
        Write computed extinction coefficients to CSV.

        Output path::

            <output_path>/analysis/extinction_coefficients/<solver>/
            extinction_coefficients_<solver>_channel_<ch>_<ref_prop>_led_array_<id>.csv
        """
        if not self.coefficients_per_image_and_layer:
            raise RuntimeError('No coefficients computed yet. Call calc_and_set_coefficients first.')

        out_path = self._output_dir()
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = (
            out_path
            / f'extinction_coefficients_{self.solver}_channel_{self.channel}'
              f'_{self.config.reference_property}_led_array_{self.led_array_id}.csv'
        )

        layers = self.layers
        sim_info = ' | '.join(
            f'sim{idx}:arr{arr}' for idx, arr in sorted(self.led_array_mapping.items())
        )
        header = (
            f'Stacked inversion | {sim_info} | solver: {self.solver} | '
            f'ref_prop: {self.config.reference_property} | '
            f'LEDSA {version("ledsa")}\n'
        )
        header += 'Layer_bottom_height[m],' + ','.join(map(str, layers.borders[:-1])) + '\n'
        header += 'Layer_top_height[m],'    + ','.join(map(str, layers.borders[1:])) + '\n'
        header += 'Layer_center_height[m],' + ','.join(
            f'{c:.4f}' for c in (layers.borders[:-1] + layers.borders[1:]) / 2
        ) + '\n'
        header += 'Experiment_Time[s],layer0'
        header += ''.join(f',layer{i + 1}' for i in range(layers.amount - 1))

        times = (
            self.common_times
            if self.common_times is not None
            else np.arange(len(self.coefficients_per_image_and_layer), dtype=float)
        )
        data = np.column_stack((times, self.coefficients_per_image_and_layer))
        np.savetxt(out_file, data, delimiter=',', header=header)
        print(f'Saved: {out_file}')

    def __str__(self) -> str:
        sim_info = ', '.join(
            f'sim{idx}(arr{arr})' for idx, arr in sorted(self.led_array_mapping.items())
        )
        return (
            f'StackedExtinctionCoefficients | led_array_id={self.led_array_id} | '
            f'channel={self.channel} | solver={self.solver} | '
            f'simulations=[{sim_info}] | '
            f'n_leds={sum(e.num_leds for e in self.sub_experiments if e.num_leds is not None)} | '
            f'n_layers={self.layers.amount}\n'
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _output_dir(self) -> Path:
        return (
            self.config.output_path
            / 'analysis'
            / 'extinction_coefficients'
            / self.solver
        )

    @staticmethod
    def _load_time_map(experiment: Experiment, average_images: bool) -> Dict[int, float]:
        info_file = (
            'image_infos_analysis_avg.csv' if average_images else 'image_infos_analysis.csv'
        )
        df = pd.read_csv(experiment.path / 'analysis' / info_file)
        return {int(i + 1): float(t) for i, t in enumerate(df['Experiment_Time[s]'])}

    @staticmethod
    def _intersect_time_axes(time_arrays: List[np.ndarray], tol: float) -> np.ndarray:
        if not time_arrays:
            return np.array([])
        common = time_arrays[0]
        for times in time_arrays[1:]:
            mask = np.array([np.any(np.abs(times - t) <= tol) for t in common])
            common = common[mask]
        return common

    @staticmethod
    def _nearest_img_id(time_map: Dict[int, float], target: float, tol: float) -> Optional[int]:
        best_id, best_diff = None, float('inf')
        for img_id, t in time_map.items():
            diff = abs(t - target)
            if diff < best_diff:
                best_diff = diff
                best_id = img_id
        return best_id if best_diff <= tol else None