# tools for analysing problems with the LED fitting

import numpy as np
import matplotlib.pyplot as plt
import importlib
from PIL import Image
import ledsa.core._led_helper as led
from ..ledsa import LEDSA


class FitAnalyser:

    def __init__(self, params):
        # params are: img_name with directory, led_id, led_line_number, fit_parameter
        #             fit_res.success, fit_res.fun, fit_res.nfev, nx, ny, im_x, im_y, window_radius
        params = params.split(' ')
        self.filename = params[0]
        self.id = int(params[1])
        self.line_number = int(params[2])
        self.fit = np.fromstring(params[3].strip('[').strip(']'), sep=",")
        self.fit_success = bool(params[4])
        self.fit_fun = params[5]
        self.fit_num_it = int(params[6])
        self.nx = int(params[7])
        self.ny = int(params[8])
        self.im_x = float(params[9])
        self.im_y = float(params[10])
        self.window_radius = int(params[11])
        self.cx = int(params[12])
        self.cy = int(params[13])
        self.channel = int(params[14])

    def plot_image(self):
        data = led.read_file(self.filename, channel=self.channel)

        s = np.index_exp[self.cx - self.window_radius:self.cx + self.window_radius,
                         self.cy - self.window_radius:self.cy + self.window_radius]
        print(s, '\n')
        im = Image.open(self.filename)
        print('%s\n', self.filename)
        im = im.crop((self.cy - 1 * self.window_radius,
                      self.cx - 1 * self.window_radius,
                      self.cy + 1 * self.window_radius,
                      self.cx + 1 * self.window_radius))
        plt.imshow(im)
        plt.show(block=False)

        mesh = np.meshgrid(np.linspace(0.5, self.nx - 0.5, self.nx), np.linspace(0.5, self.ny - 0.5, self.ny))

        led_model = led.led_fit(mesh[0], mesh[1], self.fit[0], self.fit[1], self.fit[2], self.fit[3], self.fit[4],
                                self.fit[5], self.fit[6], self.fit[7])

        fig, ax = plt.subplots(1, 2, dpi=600)

        ax[0].imshow(data[s], cmap='Greys')
        ax[0].contour(mesh[0], mesh[1], led_model, levels=10, alpha=0.3)
        ax[0].scatter(self.fit[0], self.fit[1], color='Red')

        ax[1].imshow(led_model, cmap='Greys')

        ampl = np.max(np.abs(data[s] - led_model))  # 0.25
        maxA = 255  # np.max(np.abs(data[s]))

        im2 = ax[1].imshow((data[s] - led_model)/maxA, cmap='seismic', vmin=-ampl/maxA, vmax=ampl/maxA)
        plt.colorbar(mappable=im2)
        # plt.savefig('{}_ledanalysis_{:04d}.pdf'.format(filename, iled))
        # plt.clf()

        ax[0].set_title('Fit', fontsize=3)
        ax[1].set_title('difference between fit and data \nrelative to a max luminosity of 255', fontsize=3)

        plt.show(block=True)

    def refit_image(self):
        importlib.reload(led)

        ledsa = LEDSA()
        ledsa.load_line_indices()
        ledsa.load_search_areas()
        fit_res = led.process_file(self.filename[-12:], ledsa.search_areas, ledsa.line_indices, ledsa.config['analyse_photo'], True, self.id)
        self.fit = fit_res.x
        self.fit_success = fit_res.success
        self.fit_fun = fit_res.fun
        self.fit_num_it = fit_res.nfev
