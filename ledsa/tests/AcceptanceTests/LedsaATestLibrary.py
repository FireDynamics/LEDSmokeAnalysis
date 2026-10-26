import glob
import os
import re
import shutil
import warnings
from subprocess import Popen, PIPE

import matplotlib.pyplot as plt
import numpy as np
import exiv2
from PIL import Image
from robot.api.deco import keyword, library
from robot.libraries.BuiltIn import BuiltIn
from scipy.stats import norm

from TestExperiment import TestExperiment, Layers, Camera
from ledsa.analysis.ConfigDataAnalysis import ConfigDataAnalysis
from ledsa.analysis.ConfigDataStacked import ConfigDataStacked
from ledsa.core.ConfigData import ConfigData
from ledsa.core.image_integrity import CORRUPTED_IMAGES_FILE_PATH, load_corrupted_images
from ledsa.postprocessing.simulation import SimData

# The noise put on the synthetic extinction coefficients is as large as the RMSE the tests
# accept, so drawing it afresh every run made them pass or fail by chance. A fixed seed keeps
# the test data, and with it the reconstruction error, the same on every run and platform.
NOISE_SEED = 20181127


@library
class LedsaATestLibrary:

    @keyword
    def change_dir(self, new_dir):
        os.chdir(new_dir)

    @keyword
    def create_test_data(self, num_of_leds=100, num_of_layers=20, bottom_border=0, top_border=3, background=0):
        # Create test_data directory if it doesn't exist
        if not os.path.exists('test_data'):
            os.makedirs('test_data')

        camera = Camera(0, 0, 2)
        layers = Layers(num_of_layers, bottom_border, top_border)
        extinction_coefficients_set = []

        z_range = np.linspace(bottom_border, top_border, num_of_layers)

        EPS = 1e-10  # guarantees strictly positive output
        SIGMA_NOISE = 0.05
        rng = np.random.default_rng(NOISE_SEED)

        def _add_noise(x, *, sigma=SIGMA_NOISE):
            return np.clip(x + rng.normal(0, sigma, x.shape), a_min=0.0, a_max=None) + EPS

        def extco_const_initial(z):
            # start from a tiny positive floor instead of 0 to avoid log(0)
            return np.full_like(z, EPS)

        def extco_const(z):
            return _add_noise(np.full_like(z, 0.2))

        def extco_lin(z):
            return _add_noise(0.15 * z)

        def extco_quad(z):
            return _add_noise(0.0435 * z ** 2)

        extinction_coefficients_set.append(extco_const_initial(z_range))
        extinction_coefficients_set.append(extco_const(z_range))
        extinction_coefficients_set.append(extco_lin(z_range))
        extinction_coefficients_set.append(extco_quad(z_range))

        for image_id, extinction_coefficients in enumerate(extinction_coefficients_set):
            ex = TestExperiment(camera=camera, layers=layers)
            np.savetxt(os.path.join('test_data', f'test_extinction_coefficients_input_{image_id + 1}.csv'), extinction_coefficients)
            for z in np.linspace(bottom_border + 0.05, top_border - 0.05, num_of_leds):
                ex.add_led(0, 4, z)
            ex.set_extinction_coefficients(extinction_coefficients)
            create_test_image(image_id, ex, background=int(background))

    # ------------------------------------------------------------------
    # Stacked / multi-camera keywords
    # ------------------------------------------------------------------

    @keyword
    def create_test_data_stacked(self, num_of_leds=100, num_of_layers=20, bottom_border=0, top_border=3):
        """Create test images for two cameras at different heights.

        Directory layout created under the current working directory::

            stacked/
              test_data/          ← ground-truth extinction coefficient CSVs
              cam0/test_data/     ← images as seen from the lower camera
              cam1/test_data/     ← images as seen from the upper camera
        """
        z_range = np.linspace(bottom_border, top_border, num_of_layers)

        EPS = 1e-10
        SIGMA_NOISE = 0.05
        rng = np.random.default_rng(NOISE_SEED)

        def _add_noise(x, *, sigma=SIGMA_NOISE):
            return np.clip(x + rng.normal(0, sigma, x.shape), a_min=0.0, a_max=None) + EPS

        profiles = [
            np.full_like(z_range, EPS),              # image 1 – reference (near-zero)
            _add_noise(np.full_like(z_range, 0.2)),  # image 2 – constant
            _add_noise(0.15 * z_range),              # image 3 – linear
            _add_noise(0.0435 * z_range ** 2),       # image 4 – quadratic
        ]

        # Save ground-truth extinction coefficients once (shared by both cameras)
        gt_dir = os.path.join('stacked', 'test_data')
        os.makedirs(gt_dir, exist_ok=True)
        for image_id, extco in enumerate(profiles):
            np.savetxt(
                os.path.join(gt_dir, f'test_extinction_coefficients_input_{image_id + 1}.csv'),
                extco,
            )

        # Two cameras at different heights view the same LED array
        cameras = [
            ('cam0', Camera(0, 0, 0.5)),   # lower camera
            ('cam1', Camera(0, 0, 2.5)),   # upper camera
        ]
        layers = Layers(num_of_layers, bottom_border, top_border)

        for cam_name, camera in cameras:
            img_dir = os.path.join('stacked', cam_name, 'test_data')
            os.makedirs(img_dir, exist_ok=True)
            for image_id, extco in enumerate(profiles):
                ex = TestExperiment(camera=camera, layers=layers)
                for z in np.linspace(bottom_border + 0.05, top_border - 0.05, num_of_leds):
                    ex.add_led(0, 4, z)
                ex.set_extinction_coefficients(extco)
                create_test_image(image_id, ex, img_dir=img_dir)

    @keyword
    def create_and_fill_config_for_stacked_cam(self, cam_z, first=1, last=4):
        """Create config.ini in the current directory for one camera simulation.

        The camera height *cam_z* is only needed for the LEDSA coordinate step;
        it is not stored in config.ini (that uses the physical LED array geometry).
        """
        conf = ConfigData(
            load_config_file=False,
            img_directory='test_data/',
            search_area_radius=10,
            pixel_value_percentile=99.875,
            channel=0,
            max_num_leds=1000,
            num_arrays=1,
            num_cores=1,
            date=None,
            start_time=None,
            time_ref_img_id=None,
            time_ref_img_time=None,
            time_diff_to_image_time=0,
            img_name_string='test_img_{}.jpg',
            num_img_overflow=None,
            first_img_experiment_id=first,
            last_img_experiment_id=last,
            ref_img_id=1,
            ignore_led_indices=None,
            led_array_edge_indices=None,
            led_array_edge_coordinates=None,
            first_img_analysis_id=first,
            last_img_analysis_id=last,
            num_skip_imgs=0,
            num_skip_leds=0,
            merge_led_array_indices=None,
        )
        conf.set('analyse_positions', '   led_array_edge_indices', '49 0')
        conf.set('analyse_positions', '   led_array_edge_coordinates', '0 4 0.05 0 4 2.95')
        conf.set('DEFAULT', '   date', '2018:11:27')
        conf.save()

    @keyword
    def create_and_fill_config_stacked(self, cam0_dir, cam1_dir, time_sync_tolerance=2):
        """Create config_stacked.ini in the current directory.

        *cam0_dir* and *cam1_dir* are paths (absolute or relative to the
        current working directory) to the two camera simulation directories.
        *time_sync_tolerance* is the time [s] within which images of different
        cameras are treated as simultaneous.
        """
        cfg = ConfigDataStacked(load_config_file=False)
        cfg['DEFAULT']['solver']             = 'linear'
        cfg['DEFAULT']['lambda_reg']         = '1e-3'
        cfg['DEFAULT']['reference_property'] = 'sum_col_val'
        cfg['DEFAULT']['num_ref_images']     = '1'
        cfg['DEFAULT']['ref_img_indices']    = 'None'
        cfg['DEFAULT']['average_images']     = 'False'
        cfg['DEFAULT']['camera_channels']    = '0'
        cfg['DEFAULT']['num_cores']          = '1'
        cfg['DEFAULT']['num_layers']         = '20'
        cfg['DEFAULT']['domain_bounds']      = '0 3'
        cfg['DEFAULT']['output_path']        = '.'
        cfg['DEFAULT']['time_sync_tolerance'] = str(time_sync_tolerance)

        cfg['simulation_0'] = {}
        cfg['simulation_0']['path']            = str(os.path.abspath(cam0_dir))
        cfg['simulation_0']['camera_position'] = '0 0 0.5'

        cfg['simulation_1'] = {}
        cfg['simulation_1']['path']            = str(os.path.abspath(cam1_dir))
        cfg['simulation_1']['camera_position'] = '0 0 2.5'

        cfg['led_arrays'] = {}
        cfg['led_arrays']['0'] = '0:0 1:0'

        cfg.save()

    @keyword
    def plot_stacked_input_vs_computed_extinction_coefficients(self, solver='linear', first=2, last=4, led_array_id=0, channel=0):
        """Plot input vs stacked-computed extinction coefficients for each test image."""
        filename = (
            f'extinction_coefficients_{solver}_channel_{channel}'
            f'_sum_col_val_led_array_{led_array_id}.csv'
        )
        result_path = os.path.join('analysis', 'extinction_coefficients', solver, filename)
        data = np.loadtxt(result_path, delimiter=',', skiprows=4)
        times = data[:, 0]
        extco_computed = data[:, 1:]
        num_of_layers = extco_computed.shape[1]

        os.makedirs('results', exist_ok=True)
        for image_id in range(first, last + 1):
            gt_path = os.path.join('test_data', f'test_extinction_coefficients_input_{image_id}.csv')
            extco_input = np.loadtxt(gt_path)
            row = image_id - 1

            plt.figure()
            plt.plot(extco_input, range(num_of_layers), '.-', label='Input')
            plt.plot(extco_computed[row, :], range(num_of_layers), '.-', label='Stacked computed')
            plt.xlabel(r'Extinction coefficient / m$^{-1}$')
            plt.ylabel('Layer / -')
            plt.title(f'Stacked {solver} — image {image_id}, t = {times[row]:.0f} s')
            plt.xlim(-0.05, 0.6)
            plt.ylim(-1, num_of_layers)
            plt.grid(linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join('results', f'stacked_image_{image_id}_{solver}.pdf'))
            plt.close()

    @keyword
    def check_stacked_input_vs_computed(self, image_id, solver='linear', led_array_id=0, channel=0):
        """Return the RMSE between the stacked reconstruction and the ground truth."""
        filename = (
            f'extinction_coefficients_{solver}_channel_{channel}'
            f'_sum_col_val_led_array_{led_array_id}.csv'
        )
        result_path = os.path.join('analysis', 'extinction_coefficients', solver, filename)
        data = np.loadtxt(result_path, delimiter=',', skiprows=4)
        extco_computed = data[:, 1:]   # first column is experiment time

        gt_path = os.path.join('test_data', f'test_extinction_coefficients_input_{image_id}.csv')
        extco_input = np.loadtxt(gt_path)

        row = int(image_id) - 1
        rmse = np.sqrt(np.mean((extco_input - extco_computed[row, :]) ** 2))
        return rmse

    @keyword
    def stacked_analysis_should_exclude_image(self, sim_dir, img_name, solver='linear', channel=0,
                                              led_array_id=0, reference_property='sum_col_val'):
        """Fail unless the stacked result covers every image of a simulation except the given one.

        Both camera simulations see the same images at the same experiment times, so the common time
        axis is expected to hold all of them but the time of the image that was skipped.

        :param sim_dir: Directory of the simulation the image was skipped in.
        :param img_name: Name of the image inside the test_data directory of that simulation.
        """
        infos_path = os.path.join(sim_dir, 'analysis', 'image_infos_analysis.csv')
        times_of_img = read_experiment_times_of_images(infos_path)
        if img_name not in times_of_img:
            raise AssertionError(f'{img_name} is not listed in {infos_path}')
        expected = sorted(round(t, 6) for name, t in times_of_img.items() if name != img_name)

        times, extinction_coefficients = load_stacked_extinction_coefficients_computed(
            solver, channel, led_array_id, reference_property)
        computed = [round(float(t), 6) for t in times]
        if computed != expected:
            raise AssertionError(f'Common experiment times of the stacked result are {computed}, '
                                 f'expected {expected}')
        if np.isnan(extinction_coefficients).any():
            raise AssertionError('The stacked result contains NaN extinction coefficients')

    @keyword
    def check_stacked_input_vs_computed_at_image_time(self, sim_dir, image_id, solver='linear',
                                                      led_array_id=0, channel=0):
        """Return the RMSE between the ground truth of an image and the result at its experiment time.

        The row is looked up by experiment time, because an image left out of the common time axis
        shifts every row behind it.

        :param sim_dir: Directory of the simulation the image times are read from.
        :param image_id: ID of the test image, which is also the ID of its ground truth file.
        """
        infos_path = os.path.join(sim_dir, 'analysis', 'image_infos_analysis.csv')
        img_name = f'test_img_{int(image_id)}.jpg'
        times_of_img = read_experiment_times_of_images(infos_path)
        if img_name not in times_of_img:
            raise AssertionError(f'{img_name} is not listed in {infos_path}')

        times, extco_computed = load_stacked_extinction_coefficients_computed(
            solver, channel, led_array_id)
        # The experiment times are seconds since an epoch, so they have to be compared absolutely
        rows = np.flatnonzero(np.isclose(times, times_of_img[img_name], rtol=0, atol=1e-6))
        if len(rows) != 1:
            raise AssertionError(f'{img_name} has the experiment time {times_of_img[img_name]} s, which '
                                 f'matches {len(rows)} of the computed times {list(times)}')

        gt_path = os.path.join('test_data', f'test_extinction_coefficients_input_{int(image_id)}.csv')
        extco_input = np.loadtxt(gt_path)
        return float(np.sqrt(np.mean((extco_input - extco_computed[rows[0], :]) ** 2)))

    @keyword
    def plot_input_vs_computed_extinction_coefficients(self, solver, first=1, last=4, led_array=0, channel=0):
        time, extinction_coefficients_computed = load_extinction_coefficients_computed(solver, channel, led_array)

        for image_id in range(first, last + 1):
            extinction_coefficients_input = np.loadtxt(os.path.join('test_data', f'test_extinction_coefficients_input_{image_id}.csv'), delimiter=',')
            num_of_layers = extinction_coefficients_input.shape[0]
            plt.plot(extinction_coefficients_input, range(0, num_of_layers), '.-', label='Input')
            plt.plot(extinction_coefficients_computed[image_id - 1, :], range(0, num_of_layers), '.-', label='Computed')
            plt.xlabel('Extinction coefficient / $\mathrm{m}^{-1}$')
            plt.ylabel('Layer / -')
            plt.title(f'Input vs Computed {solver} Extinction Coefficients - Image {image_id}, t = {time[image_id - 1]} s')
            plt.xlim(-0.1, 0.6)
            plt.ylim(0, num_of_layers)
            plt.grid(linestyle='--', alpha=0.5)
            plt.legend()
            if not os.path.exists('results'):
                os.makedirs('results')
            plt.savefig(os.path.join('results', f'image_Id_{image_id}_{solver}.pdf'))
            plt.close()

    @keyword
    def check_input_vs_computed_extinction_coefficients(self, image_id, solver, led_array=0, channel=0,
                                                        reference_property='sum_col_val'):
        _, extinction_coefficients_computed = load_extinction_coefficients_computed(solver, channel, led_array,
                                                                                    reference_property)
        extinction_coefficients_input = np.loadtxt(os.path.join('test_data', f'test_extinction_coefficients_input_{image_id}.csv'), delimiter=',')
        rmse = np.sqrt(
            np.mean((extinction_coefficients_input - extinction_coefficients_computed[int(image_id) - 1, :]) ** 2))
        return rmse

    @keyword
    def create_and_fill_config(self, first=1, last=4):
        # Create test_data directory if it doesn't exist
        if not os.path.exists('test_data'):
            os.makedirs('test_data')

        conf = ConfigData(load_config_file=False, img_directory='test_data/', search_area_radius=10, pixel_value_percentile=99.875,
                          channel=0, max_num_leds=1000, num_arrays=1, num_cores=1, date=None,
                          start_time=None, time_ref_img_id=None, time_ref_img_time=None, time_diff_to_image_time=0,
                          img_name_string='test_img_{}.jpg', num_img_overflow=None, first_img_experiment_id=first,
                          last_img_experiment_id=last, ref_img_id=1, ignore_led_indices=None,
                          led_array_edge_indices=None, led_array_edge_coordinates=None,
                          first_img_analysis_id=first, last_img_analysis_id=last, num_skip_imgs=0, num_skip_leds=0,
                          merge_led_array_indices=None)
        conf.set('analyse_positions', '   led_array_edge_indices', '49 0')
        conf.set('analyse_positions', '   led_array_edge_coordinates', '0 4 0.05 0 4 2.95')
        conf.set('DEFAULT', '   date', '2018:11:27')
        conf.save()

    @keyword
    def create_and_fill_config_analysis(self, solver, reference_property='sum_col_val'):
        conf = ConfigDataAnalysis(load_config_file=False, camera_position=None, num_layers=20, domain_bounds=None,
                                  led_array_indices=0, num_ref_images=1, camera_channels=0, num_cores=1,
                                  reference_property=reference_property,
                                  average_images=False, solver=solver, weighting_preference=-6e-4,
                                  weighting_curvature=1e-7,
                                  num_iterations=2000, lambda_reg=1e-3)
        conf.set('experiment_geometry', '   camera_position', '0 0 2')
        conf.set('model_parameters', '   domain_bounds', '0 3')
        conf.save()

    @keyword
    def execute_ledsa_s1(self, use_config):
        if use_config:
            out = self.execute_ledsa('-s1')
        else:
            self.execute_ledsa('--config')
            inp = b'test_data/\ntest_img_{}.jpg\n1\n12:00:00\n1\n1000\n1\n1\n1'
            out = self.execute_ledsa('-s1', inp)
            check_error_msg(out)
        stdout = out[0].decode('utf-8')
        leds_found = re.search(r'Found (\d+) LEDS', stdout)
        if leds_found is None:
            raise AssertionError(f'step 1 did not report a number of LEDs: {stdout[-200:]!r}')
        return leds_found.group(1)

    @keyword
    def execute_ledsa(self, arg, inp=None):
        p = Popen(['python', '-m', 'ledsa', arg], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out = wait_for_process_to_finish(p, inp)
        return out

    @keyword
    def create_cc_matrix_file(self):
        file = open("mean_all_cc_matrix_integral.csv", "w", encoding="utf-8")
        file.write("2,3,4\n1,2,7\n3,4,5")
        file.close()

    @keyword
    def strip_bgsub_column_from_led_position_files(self, channel=0, num_columns=5):
        """Keep only the first num_columns columns of the step 3 output files to simulate data extracted with
        an older ledsa version (5: prior to the local background subtraction). The column header is shortened
        accordingly."""
        num_columns = int(num_columns)
        pattern = os.path.join('analysis', f'channel{channel}', '*_led_positions.csv')
        for path in glob.glob(pattern):
            with open(path, encoding='utf-8') as file:
                lines = file.readlines()
            with open(path, 'w', encoding='utf-8') as file:
                for line in lines:
                    if line.startswith('# id,line,'):
                        file.write(','.join(line.strip().split(',')[:num_columns]) + '\n')
                    elif not line.strip() or line.lstrip().startswith('#'):
                        file.write(line)
                    else:
                        file.write(','.join(line.strip().split(',')[:num_columns]) + '\n')

    @keyword
    def check_extinction_coefficient_files_are_identical(self, file_a, file_b):
        data_a = np.loadtxt(file_a, delimiter=',')
        data_b = np.loadtxt(file_b, delimiter=',')
        return bool(np.array_equal(data_a, data_b))

    @keyword
    def corrupt_image(self, img_name, mode='truncate'):
        """Damage a test image and keep a backup, so that it can be restored afterwards.

        :param img_name: Name of the image inside the test_data directory.
        :param mode: 'truncate' cuts the file in half, which leaves the EXIF metadata readable but breaks
                     the pixel data. 'garbage' replaces the content, which breaks both.
        """
        img_path = os.path.join('test_data', img_name)
        backup_path = img_path + '.backup'
        if not os.path.exists(backup_path):
            shutil.copyfile(img_path, backup_path)
        if mode == 'truncate':
            content = open(backup_path, 'rb').read()
            with open(img_path, 'wb') as img_file:
                img_file.write(content[:len(content) // 2])
        elif mode == 'garbage':
            with open(img_path, 'wb') as img_file:
                img_file.write(b'this is not an image' * 20)
        else:
            raise ValueError(f'Unknown corruption mode {mode}')

    @keyword
    def restore_corrupted_images(self):
        """Restore all images damaged by 'Corrupt Image' and remove the report of the corrupted images."""
        for backup_path in glob.glob(os.path.join('test_data', '*.backup')):
            shutil.move(backup_path, backup_path[:-len('.backup')])
        if os.path.exists(CORRUPTED_IMAGES_FILE_PATH):
            os.remove(CORRUPTED_IMAGES_FILE_PATH)

    @keyword
    def get_corrupted_images_report(self):
        """Read the report of the corrupted images as a list of dictionaries."""
        return load_corrupted_images()

    @keyword
    def corrupted_images_report_should_list(self, img_name, analysis_step):
        """Fail unless the report lists the image for the given analysis step.

        :param img_name: Name of the image inside the test_data directory.
        :param analysis_step: Analysis step that is expected to have skipped the image.
        """
        rows = load_corrupted_images()
        img_path = os.path.join('test_data', img_name)
        for row in rows:
            if (os.path.normpath(row['image_path']) == os.path.normpath(img_path)
                    and row['analysis_step'] == analysis_step):
                if not row['error_type'] or not row['error_message']:
                    raise AssertionError(f'{img_path} is listed without a reason: {row}')
                return
        raise AssertionError(f'{img_path} is not listed for "{analysis_step}". Report contains: {rows}')

    @keyword
    def analysis_should_exclude_image(self, img_name, solver='linear', channel=0, led_array=0,
                                      reference_property='sum_col_val'):
        """Fail unless the extinction coefficients cover every analysed image except the given one.

        Checks that the skipped image contributes no row and, more importantly, that the remaining rows are
        still paired with the experiment time of their own image instead of being shifted by one.
        """
        infos_path = os.path.join('analysis', 'image_infos_analysis.csv')
        times_of_img = read_experiment_times_of_images(infos_path)
        if img_name not in times_of_img:
            raise AssertionError(f'{img_name} is not listed in {infos_path}')

        filename = (f'extinction_coefficients_{solver}_channel_{channel}_{reference_property}'
                    f'_led_array_{led_array}.csv')
        file_path = os.path.join('analysis', 'extinction_coefficients', solver, filename)
        data = np.atleast_2d(np.genfromtxt(file_path, delimiter=',', comments='#'))
        times = [round(t, 6) for t in data[:, 0]]

        expected = sorted(round(t, 6) for name, t in times_of_img.items() if name != img_name)
        if times != expected:
            raise AssertionError(f'Experiment times in {filename} are {times}, expected {expected}')
        if np.isnan(data[:, 1:]).any():
            raise AssertionError(f'{filename} contains NaN extinction coefficients')

    @keyword
    def extco_time_axis_should_be_accepted(self):
        """Fail if reading the extinction coefficients warns about the experiment times of their files."""
        messages = _read_extco_with_warnings()
        if messages:
            raise AssertionError(f'Reading the extinction coefficients warned: {messages}')

    @keyword
    def extco_time_axis_mismatch_should_be_reported(self, shift=0.5):
        """Fail unless reading the extinction coefficients warns once the image times have moved.

        Shifts every experiment time in the image infos, which is what re-running step 1 or step 3 with
        other image metadata does to them, and puts the file back afterwards.

        :param shift: Seconds added to every experiment time.
        """
        infos_path = os.path.join('analysis', 'image_infos_analysis.csv')
        with open(infos_path, encoding='utf-8') as infos_file:
            original = infos_file.read()
        shifted_lines = []
        for line in original.splitlines():
            if line.startswith('#') or not line.strip():
                shifted_lines.append(line)
                continue
            img_id, name, clock, experiment_time = line.split(',')
            shifted_lines.append(f'{img_id},{name},{clock},{float(experiment_time) + float(shift)}')
        try:
            with open(infos_path, 'w', encoding='utf-8') as infos_file:
                infos_file.write('\n'.join(shifted_lines) + '\n')
            messages = _read_extco_with_warnings()
        finally:
            with open(infos_path, 'w', encoding='utf-8') as infos_file:
                infos_file.write(original)
        if not any('Run the analysis again' in message for message in messages):
            raise AssertionError('Reading the extinction coefficients did not report that they were '
                                 f'computed for other experiment times. Warnings: {messages}')

    @keyword
    def get_analysis_image_id(self, img_name):
        """Get the analysis ID that 'image_infos_analysis.csv' assigns to an image."""
        file_path = os.path.join('analysis', 'image_infos_analysis.csv')
        with open(file_path, encoding='utf-8') as infos_file:
            for line in infos_file:
                if line.startswith('#') or not line.strip():
                    continue
                img_id, name = line.split(',')[:2]
                if name == img_name:
                    return img_id
        raise AssertionError(f'{img_name} is not listed in {file_path}')

    @keyword
    def rename_test_images_to_padded_ids(self, num_digits=4):
        """Rename test_data/test_img_<id>.jpg to zero padded IDs (test_img_0001.jpg), as most cameras write them."""
        for path in glob.glob(os.path.join('test_data', 'test_img_*.jpg')):
            img_id = int(os.path.basename(path)[len('test_img_'):-len('.jpg')])
            os.rename(path, os.path.join('test_data', f'test_img_{img_id:0{int(num_digits)}d}.jpg'))

    @keyword
    def set_config_option(self, section, option, value):
        """Set an option in the config.ini of the current directory."""
        conf = ConfigData()
        conf[section][option] = str(value)
        conf.save()

    @keyword
    def analysed_image_names_should_be(self, *expected_img_names):
        """Fail unless exactly the given images are listed in 'image_infos_analysis.csv'."""
        file_path = os.path.join('analysis', 'image_infos_analysis.csv')
        img_names = []
        with open(file_path, encoding='utf-8') as infos_file:
            for line in infos_file:
                if not line.startswith('#') and line.strip():
                    img_names.append(line.split(',')[1])
        if img_names != list(expected_img_names):
            raise AssertionError(f'Analysed images are {img_names}, expected {list(expected_img_names)}')

def read_experiment_times_of_images(infos_path):
    """Read the experiment times of an 'image_infos_analysis.csv' file as {image name: time}."""
    times_of_img = {}
    with open(infos_path, encoding='utf-8') as infos_file:
        for line in infos_file:
            if line.startswith('#') or not line.strip():
                continue
            _, name, _, experiment_time = line.strip().split(',')
            times_of_img[name] = float(experiment_time)
    return times_of_img

def load_stacked_extinction_coefficients_computed(solver, channel, led_array_id,
                                                  reference_property='sum_col_val'):
    filename = (f'extinction_coefficients_{solver}_channel_{channel}'
                f'_{reference_property}_led_array_{led_array_id}.csv')
    data = np.atleast_2d(np.loadtxt(
        os.path.join('analysis', 'extinction_coefficients', solver, filename), delimiter=','))
    return data[:, 0], data[:, 1:]

def load_extinction_coefficients_computed(solver, channel, led_array, reference_property='sum_col_val'):
    filename = f'extinction_coefficients_{solver}_channel_{channel}_{reference_property}_led_array_{led_array}.csv'
    data = np.loadtxt(
        os.path.join('analysis', 'extinction_coefficients', solver, filename),delimiter=',')
    time = data[:, 0]
    extinction_coefficients_computed = data[:, 1:]
    return time, extinction_coefficients_computed

def create_test_image(image_id, experiment, img_dir='test_data', background=0):
    """ Creates three test images with black and gray pixels representing 3 leds and sets the exif data needed
    The first image has 100% transmission on all LEDs, the second image has 50% transmission on all LEDs,
    the third has 50%, 70% and 80% transmission on the top, middle and bottom LEDs.
    :return: None
    """
    # Create img_dir directory if it doesn't exist
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    num_of_leds = len(experiment.leds)
    transmissions = experiment.calc_all_led_transmissions()

    # Reverse transmissions because images are created from top down
    img_array = create_img_array(num_of_leds, list(reversed(transmissions)), background=background)
    img = Image.fromarray(img_array, 'RGB')

    # Save image without EXIF data
    out = os.path.join(img_dir, f"test_img_{image_id + 1}.jpg")
    img.save(out)

    # Add EXIF data to the image afterward
    img2 = exiv2.ImageFactory.open(out)
    img2.readMetadata()
    ex = img2.exifData()
    ex["Exif.Photo.DateTimeOriginal"] = f"2021:01:01 12:00:{image_id:02d}"
    # Cameras store the fraction of the second in a tag of its own. A different fraction per image makes
    # sure that it reaches the image times instead of being rounded away.
    ex["Exif.Photo.SubSecTimeOriginal"] = f"{image_id * 25 % 100:02d}"
    img2.setExifData(ex)
    img2.writeMetadata()


def create_img_array(num_of_leds, transmissions, background=0):
    img = np.zeros((num_of_leds * 50 + 50, 50, 3), np.uint8)
    for led_id in range(num_of_leds):
        add_led(img, (1 + led_id) * 50, 25, transmissions[led_id])
    if background:
        img = np.clip(img.astype(np.int16) + int(background), 0, 255).astype(np.uint8)
    return img


def add_led(img, x_pos, y_pos, transmission):
    rv = norm()
    size = 20
    led = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            led[x, y] = calc_color_val(x, y, size, rv) * transmission
    img[x_pos - size // 2:x_pos + size // 2, y_pos - size // 2:y_pos + size // 2, 0] = led
    img[:, :, 1] = img[:, :, 0]
    img[:, :, 2] = img[:, :, 0]


def calc_color_val(x, y, size, rv):
    dist = ((size / 2 - x) ** 2 + (size / 2 - y) ** 2) ** 0.5
    scale = 1.7
    return rv.pdf(dist / scale) * 350 * scale


def _read_extco_with_warnings():
    """Read the extinction coefficient files through the postprocessing and return its warnings."""
    experiment_dir = os.getcwd()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            sim_data = SimData(experiment_dir, read_all=False, load_config_params=True)
            sim_data._get_extco_df_from_path()
    finally:
        # SimData changes into the experiment directory and does not change back
        os.chdir(experiment_dir)
    return [str(entry.message) for entry in caught if entry.category is UserWarning]


def wait_for_process_to_finish(p, inp=None):
    out = p.communicate(inp)
    check_error_msg(out)
    return out


def check_error_msg(out):
    if out[1] is not None:
        stderr_output = out[1].decode('utf-8')
        # Filter out tqdm progress bar output
        if stderr_output and not is_tqdm_output(stderr_output):
            BuiltIn().log(stderr_output, 'ERROR')
            exit()

def is_tqdm_output(text):
    """
    Check if the text is likely a tqdm progress bar output.

    :param text: Text to check
    :return: True if the text appears to be from tqdm, False otherwise
    """

    return "Processing images" in text