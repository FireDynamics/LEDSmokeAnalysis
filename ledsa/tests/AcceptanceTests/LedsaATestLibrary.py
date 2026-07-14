import glob
import os
from subprocess import Popen, PIPE

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
import exiv2
from PIL import Image
from robot.api.deco import keyword, library
from robot.libraries.BuiltIn import BuiltIn
from scipy.stats import norm

from TestExperiment import TestExperiment, Layers, Camera
from ledsa.analysis.ConfigDataAnalysis import ConfigDataAnalysis
from ledsa.analysis.ConfigDataStacked import ConfigDataStacked
from ledsa.core.ConfigData import ConfigData


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

        def _add_noise(x, *, sigma=SIGMA_NOISE):
            return np.clip(x + normal(0, sigma, x.shape), a_min=0.0, a_max=None) + EPS

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

        def _add_noise(x, *, sigma=SIGMA_NOISE):
            return np.clip(x + normal(0, sigma, x.shape), a_min=0.0, a_max=None) + EPS

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
    def create_and_fill_config_stacked(self, cam0_dir, cam1_dir):
        """Create config_stacked.ini in the current directory.

        *cam0_dir* and *cam1_dir* are paths (absolute or relative to the
        current working directory) to the two camera simulation directories.
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
        cfg['DEFAULT']['time_sync_tolerance'] = '2'

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
        return out[0].decode('utf-8')[-9:-6]

    @keyword
    def execute_ledsa(self, arg, inp=None):
        p = Popen(['python', '-m', 'ledsa', arg], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out = wait_for_process_to_finish(p, inp)
        return out

    @keyword
    def create_cc_matrix_file(self):
        file = open("mean_all_cc_matrix_integral.csv", "w")
        file.write("2,3,4\n1,2,7\n3,4,5")
        file.close()

    @keyword
    def strip_bgsub_column_from_led_position_files(self, channel=0):
        """Remove the bgsub_sum_col_value column from the step 3 output files to simulate
        data extracted with a ledsa version prior to the local background subtraction."""
        pattern = os.path.join('analysis', f'channel{channel}', '*_led_positions.csv')
        for path in glob.glob(pattern):
            with open(path) as file:
                lines = file.readlines()
            with open(path, 'w') as file:
                for line in lines:
                    if not line.strip() or line.lstrip().startswith('#'):
                        file.write(line)
                    else:
                        file.write(','.join(line.strip().split(',')[:5]) + '\n')

    @keyword
    def check_extinction_coefficient_files_are_identical(self, file_a, file_b):
        data_a = np.loadtxt(file_a, delimiter=',')
        data_b = np.loadtxt(file_b, delimiter=',')
        return bool(np.array_equal(data_a, data_b))

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