# tools for analysing problems with the LED fitting

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ..core._led_helper import read_file
from ..core._led_helper import led_fit


class FitAnalyser:

    def __init__(self, params):
        # params are: img_name with directory, led_id, led_line_number, fit_parameter
        #             fit_res.success, fit_res.fun, fit_res.nfev, nx, ny, im_x, im_y, window_radius
        params = params.split(' ')
        self.filename = params[0]
        self.id = int(params[1])
        self.line_number = int(params[2])
        self.fit = np.array(params[3])
        self.fit_success = bool(params[4])
        self.fit_fun = params[5]
        self.fit_num_it = params[6]
        self.nx = params[7]
        self.ny = params[8]
        self.im_x = params[9]
        self.im_y = params[10]
        self.window_radius = float(params[11])

    def plot_image(self):
        im = Image.open(self.filename)
        im = im.crop((self.im_x - 1.5 * self.window_radius,
                      self.im_y + 1.5 * self.window_radius,
                      self.im_x + 1.5 * self.window_radius,
                      self.im_y - 1.5 * self.window_radius))

        plt.figure(dpi=1200)

        mesh = np.meshgrid(np.linspace(0.5, nx - 0.5, nx), np.linspace(0.5, ny - 0.5, ny))

        led_model = led_fit(mesh[0], mesh[1], self.fit[0], self.fit[1], self.fit[2], self.fit[3], self.fit[4],
                            self.fit[5], self.fit[6], self.fit[7])

        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(im, cmap='Greys')
        ax[0].contour(mesh[0], mesh[1], led_model, levels=10, alpha=0.3)
        ax[0].scatter(self.fit[0], self.fit[1], color='Red')

        # ax[1].imshow(led_model, cmap='Greys')

        ampl = 0.25 # np.max(np.abs(data[s] - led_model))

        #im2 = ax[1].imshow((data[s] - led_model)/maxA, cmap='seismic', vmin=-ampl, vmax=ampl)
        plt.colorbar(mappable=im2)
        plt.show()
        # plt.savefig('{}_ledanalysis_{:04d}.pdf'.format(filename, iled))
        # plt.clf()

        plt.imshow(im, cmap='Greys')
        plt.colorbar()
        plt.plot()
