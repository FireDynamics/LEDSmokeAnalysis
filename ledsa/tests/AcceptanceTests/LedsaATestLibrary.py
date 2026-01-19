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
from ledsa.core.ConfigData import ConfigData


@library
class LedsaATestLibrary:

    @keyword
    def change_dir(self, new_dir):
        os.chdir(new_dir)

    @keyword
    def create_test_data(self, num_of_leds=100, num_of_layers=20, bottom_border=0, top_border=3):
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
            create_test_image(image_id, ex)

    @keyword
    def plot_input_vs_computed_extinction_coefficients(self, solver, first=1, last=4, led_array=0, channel=0):
        filename = f'extinction_coefficients_{solver}_channel_{channel}_sum_col_val_led_array_{led_array}.csv'
        extinction_coefficients_computed = (
            np.loadtxt(os.path.join('analysis', 'extinction_coefficients', solver, filename), skiprows=5, delimiter=','))
        for image_id in range(first, last + 1):
            extinction_coefficients_input = np.loadtxt(os.path.join('test_data', f'test_extinction_coefficients_input_{image_id}.csv'), delimiter=',')
            num_of_layers = extinction_coefficients_input.shape[0]
            plt.plot(extinction_coefficients_input, range(num_of_layers, 0, -1), '.-', label='Input')
            plt.plot(extinction_coefficients_computed[image_id - 1, :], range(num_of_layers, 0, -1), '.-', label='Computed')
            plt.xlabel('Extinction coefficient / $\mathrm{m}^{-1}$')
            plt.ylabel('Layer / -')
            plt.title(f'Input vs Computed {solver} Extinction Coefficients - Image {image_id}')
            plt.xlim(-0.1, 0.6)
            plt.ylim(num_of_layers, 0)
            plt.grid(linestyle='--', alpha=0.5)
            plt.legend()
            if not os.path.exists('results'):
                os.makedirs('results')
            plt.savefig(os.path.join('results', f'image_Id_{image_id}_{solver}.pdf'))
            plt.close()

    @keyword
    def check_input_vs_computed_extinction_coefficients(self, image_id, solver, led_array=0, channel=0):
        filename = f'extinction_coefficients_{solver}_channel_{channel}_sum_col_val_led_array_{led_array}.csv'
        extinction_coefficients_computed = (
            np.loadtxt(os.path.join('analysis', 'extinction_coefficients',solver, filename), skiprows=5, delimiter=','))
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
                          start_time=None, time_img_id=None, time_ref_img_time=None, time_diff_to_image_time=0,
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
    def create_and_fill_config_analysis(self, solver):
        conf = ConfigDataAnalysis(load_config_file=False, camera_position=None, num_layers=20, domain_bounds=None,
                                  led_array_indices=0, num_ref_images=1, camera_channels=0, num_cores=1,
                                  reference_property='sum_col_val',
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


def create_test_image(image_id, experiment):
    """ Creates three test images with black and gray pixels representing 3 leds and sets the exif data needed
    The first image has 100% transmission on all LEDs, the second image has 50% transmission on all LEDs,
    the third has 50%, 70% and 80% transmission on the top, middle and bottom LEDs.
    :return: None
    """
    # Create test_data directory if it doesn't exist
    if not os.path.exists('test_data'):
        os.makedirs('test_data')

    num_of_leds = len(experiment.leds)
    transmissions = experiment.calc_all_led_transmissions()
    img_array = create_img_array(num_of_leds, transmissions)
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array, 'RGB')

    # Save image without EXIF data
    out = os.path.join("test_data", f"test_img_{image_id + 1}.jpg")
    img.save(out)

    # Add EXIF data to the image afterward
    img2 = exiv2.ImageFactory.open(out)
    img2.readMetadata()
    ex = img2.exifData()
    ex["Exif.Photo.DateTimeOriginal"] = f"2021:01:01 12:00:{image_id:02d}"
    img2.setExifData(ex)
    img2.writeMetadata()


def create_img_array(num_of_leds, transmissions):
    img = np.zeros((num_of_leds * 50 + 50, 50, 3), np.int8)
    for led_id in range(num_of_leds):
        add_led(img, (1 + led_id) * 50, 25, transmissions[led_id])
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
