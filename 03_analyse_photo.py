import numpy as np
import matplotlib.pyplot as plt

import led_helper as led

root_directory = '2018.11/V1_C1'

data_indices = [7460 + 50*i for i in range(10)]

def process_file(idata):
    
    img_filename = 'IMG_{}.JPG'.format(idata)

    search_areas = np.loadtxt('out/{}/led_search_areas.csv'.format(root_directory))
    print(search_areas)
    print(len(search_areas))

    data = led.read_file('data/{}/{}'.format(root_directory, img_filename), channel=0)
    print(data)

    out_file = open('out/{}/{}.led_positions.csv'.format(root_directory,img_filename), 'w')
    out_file.write("# id,         line,   x,         y,        dx,        dy,         A,     alpha,        wx,        wy, fit_success,   fit_fun, fit_nfev // all spatial quatities in pixel coordinates\n")
    window_radius = 10

    nled = search_areas.shape[0]
    # nled = 10
    for iline in range(7):
    #for iline in [3]:
        line_indices = np.loadtxt('out/{}/line_indices_{:03d}.csv'.format(root_directory, iline),
                   delimiter=',', dtype='int')
        for iled in line_indices:
        # selected_indices = [line_indices[12], line_indices[50], line_indices[140]]
        # for iled in selected_indices:
            # print("processing led {:4d} out of {:4d}".format(iled, nled))

            cx = int(search_areas[iled,1])
            cy = int(search_areas[iled,2])

            print(cx, cy)

            s = np.index_exp[cx - window_radius:cx + window_radius,
                             cy - window_radius:cy + window_radius]

            maxA = np.max(data[s])

            fit_res, mesh = led.find_leds(data[s])

            x, y, dx, dy, A, alpha, wx, wy = fit_res.x

            im_x = x + cx - window_radius
            im_y = y + cy - window_radius

            line_number = iline

            out_file.write('{:4d}, {:2d}, {:10.4e}, {:10.4e}, {:10.4e}, {:10.4e}, {:10.4e}, {:10.4e}, {:10.4e}, {:10.4e}, {:12d}, {:10.4e}, {:9d}\n'.format(iled, line_number, im_x, im_y, dx, dy, A, alpha, wx, wy, fit_res.success, fit_res.fun, fit_res.nfev))

    out_file.close()

from multiprocessing import Pool

with Pool(4) as p:
    p.map(process_file, data_indices)
