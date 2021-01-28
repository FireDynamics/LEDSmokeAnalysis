from robot.api.deco import keyword, library
from robot.libraries.BuiltIn import BuiltIn
import os
try:
    from PIL import Image
except ImportError:
    import Image
import numpy as np
from scipy.stats import norm
from ledsa.core.ConfigData import ConfigData
from subprocess import Popen, PIPE
import piexif


@library
class LedsaATestLibrary:

    @keyword
    def change_dir(self, new_dir):
        os.chdir(new_dir)

    @keyword
    def create_test_image(self, amount=1):
        img_array = create_img_array()
        for i in range(amount):
            img = Image.fromarray(img_array, 'RGB')
            exif_ifd = {
                piexif.ExifIFD.DateTimeOriginal: u'2021:01:01 12:00:00'
            }
            exif_dict = {'Exif': exif_ifd}
            exif_bytes = piexif.dump(exif_dict)
            img.save(f'test_img_{i}.jpg', exif=exif_bytes)

    @keyword
    def create_and_fill_config(self):
        conf = ConfigData(False, '.', 10, 0.25, 1, False, 1, 'test_img_0.jpg', None, None, 0, None, 'test_img_{}.jpg',
                          0, 0, 0, 0)
        conf.set('analyse_positions', '   line_edge_indices', '0 2')
        conf.set('DEFAULT', '   date', '2018:11:27')
        conf.save()

    @keyword
    def execute_ledsa_s1(self, use_config):
        if use_config:
            out = self.execute_ledsa('-s1')
        else:
            self.execute_ledsa('--config')
            inp = b'test_img_0.jpg\ntest_img_0.jpg\n12:00:00\ntest_img_{}.jpg\n0\n0\n1\n'
            out = self.execute_ledsa('-s1', inp)
            check_error_msg(out)
        return out[0].decode('ascii')[-7]

    @keyword
    def execute_ledsa(self, arg, inp=None):
        p = Popen(['python', '-m', 'ledsa', arg], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out = wait_for_process_to_finish(p, inp)
        return out


def create_img_array():
    img = np.zeros((200, 50, 3), np.int8)
    add_led(img, 50, 25)
    add_led(img, 100, 25)
    add_led(img, 150, 25)
    return img


def add_led(img, x_pos, y_pos):
    rv = norm()
    size = 20
    led = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            led[x, y] = calc_color_val(x, y, size, rv)
    img[x_pos - size//2:x_pos + size//2, y_pos - size//2:y_pos + size//2, 0] = led
    img[:, :, 1] = img[:, :, 0]
    img[:, :, 2] = img[:, :, 0]


def calc_color_val(x, y, size, rv):
    dist = ((size/2 - x)**2 + (size/2 - y)**2)**0.5
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
