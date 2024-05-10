import os
from subprocess import Popen, PIPE

import matplotlib.pyplot as plt
import numpy as np
import piexif
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
        camera = Camera(0, 0, 2)
        layers = Layers(num_of_layers, bottom_border, top_border)

        extinction_coefficients_set = []
        extinction_coefficients_set.append(np.zeros(num_of_layers))
        extinction_coefficients_set.append(0.2 * np.ones(num_of_layers))

        z_range = np.linspace(bottom_border, top_border, num_of_layers)

        def extco_lin(z):
            return 0.15 * z

        def extco_quad(z):
            return 0.0435 * z ** 2

        extinction_coefficients_set.append(extco_lin(z_range))
        extinction_coefficients_set.append(extco_quad(z_range))

        for image_id, extinction_coefficients in enumerate(extinction_coefficients_set):
            ex = TestExperiment(camera=camera, layers=layers)
            np.savetxt(f'test_extinction_coefficients_input_{image_id + 1}.csv', extinction_coefficients)
            for z in np.linspace(bottom_border + 0.05, top_border - 0.05, num_of_leds):
                ex.add_led(0, 4, z)
            ex.set_extinction_coefficients(extinction_coefficients)
            create_test_image(image_id, ex)

    @keyword
    def plot_input_vs_computed_extinction_coefficients(self, first=1, last=4, led_array=0, channel=0):
        filename = f'absorption_coefs_numeric_channel_{channel}_sum_col_val_led_array_{led_array}.csv'
        extinction_coefficients_computed = (
            np.loadtxt(os.path.join('analysis', 'AbsorptionCoefficients', filename), skiprows=5, delimiter=','))
        for image_id in range(first, last + 1):
            extinction_coefficients_input = np.flip(
                np.loadtxt(f'test_extinction_coefficients_input_{image_id}.csv', delimiter=','))
            num_of_layers = extinction_coefficients_input.shape[0]
            plt.plot(extinction_coefficients_input, range(num_of_layers), '.-')
            plt.plot(extinction_coefficients_computed[image_id - 1, :], range(num_of_layers), '.-')
            plt.xlim(-0.1, 0.8)
            plt.ylim(num_of_layers, 0)
            plt.grid(linestyle='--', alpha=0.5)
            plt.savefig(f'image_Id_{image_id}.pdf')
            plt.close()

    @keyword
    def check_input_vs_computed_extinction_coefficients(self, image_id, led_array=0, channel=0):
        filename = f'absorption_coefs_numeric_channel_{channel}_sum_col_val_led_array_{led_array}.csv'
        extinction_coefficients_computed = (
            np.loadtxt(os.path.join('analysis', 'AbsorptionCoefficients', filename), skiprows=5, delimiter=','))
        extinction_coefficients_input = np.flip(
            np.loadtxt(f'test_extinction_coefficients_input_{image_id}.csv', delimiter=','))
        rmse = np.sqrt(
            np.mean((extinction_coefficients_input - extinction_coefficients_computed[int(image_id) - 1, :]) ** 2))
        return rmse

    @keyword
    def create_and_fill_config(self, first=1, last=4):
        conf = ConfigData(load_config_file=False, img_directory='./', window_radius=10, pixel_value_percentile=99.875,
                          channel='all', max_num_of_leds=1000, num_of_arrays=1, num_of_cores=1, date=None,
                          start_time=None, time_img=None, time_ref_img_time=None, time_diff_to_image_time=0,
                          img_name_string='test_img_{}.jpg', img_number_overflow=None, first_img_experiment=first,
                          last_img_experiment=last, reference_img='test_img_1.jpg', ignore_indices=None,
                          line_edge_indices=None, line_edge_coordinates=None, first_img_analysis=first,
                          last_img_analysis=last, skip_imgs=0, skip_leds=0, merge_led_arrays=None)
        conf.set('analyse_positions', '   line_edge_indices', '0 99')
        conf.set('analyse_positions', '   line_edge_coordinates', '0 4 0.05 0 4 2.95')
        conf.set('DEFAULT', '   date', '2018:11:27')
        conf.save()

    @keyword
    def create_and_fill_config_analysis(self):
        conf = ConfigDataAnalysis(load_config_file=False, camera_position=None, num_of_layers=20, domain_bounds=None,
                                  led_arrays=0, num_ref_images=1, camera_channels=0, num_of_cores=1,
                                  reference_property='sum_col_val',
                                  average_images=False, solver='numeric', weighting_preference=-6e-3,
                                  weighting_curvature=1e-6,
                                  num_iterations=2000)
        conf.set('experiment_geometry', '   camera_position', '0 0 2')
        conf.set('model_parameters', '   domain_bounds', '0 3')
        conf.save()

    @keyword
    def execute_ledsa_s1(self, use_config):
        if use_config:
            out = self.execute_ledsa('-s1')
        else:
            self.execute_ledsa('--config')
            inp = b'./\ntest_img_1.jpg\ntest_img_1.jpg\n12:00:00\n1\n1\n1'
            out = self.execute_ledsa('-s1', inp)
            check_error_msg(out)
        return out[0].decode('ascii')[-9:-6]

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
    num_of_leds = len(experiment.leds)
    transmissions = experiment.calc_all_led_transmissions()
    img_array = create_img_array(num_of_leds, transmissions)

    img = Image.fromarray(img_array, 'RGB')
    exif_ifd = {
        piexif.ExifIFD.DateTimeOriginal: f'2021:01:01 12:00:{0 + image_id:01d}'
    }
    exif_dict = {'Exif': exif_ifd}
    exif_bytes = piexif.dump(exif_dict)
    img.save(f'test_img_{image_id + 1}.jpg', exif=exif_bytes)


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
        if out[1].decode('ascii') != '':
            BuiltIn().log(out[1].decode('ascii'), 'ERROR')
            exit()

# ddd = LedsaATestLibrary()
# ddd.create_test_data()
# ddd.plot_input_vs_computed_extinction_coefficients()
