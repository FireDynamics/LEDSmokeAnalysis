import numpy as np
import sys
import matplotlib.pyplot as plt

import led_helper as led

root_directory = '2018.11/V1_C1'
img_filename = 'IMG_7460.JPG'
out_filename = 'out/{}/led_search_areas.csv'.format(root_directory)

filename = "data/{}/{}".format(root_directory, img_filename)
data = led.read_file(filename, channel=0)
window_radius=10
xys = led.find_search_areas(data, window_radius=window_radius, skip=1)

np.savetxt(out_filename, xys,
           header='LED id, pixel position x, pixel position y', fmt='%d')

fig = plt.figure(dpi=1200)
ax = plt.gca()

for i in range(xys.shape[0]):
    ax.add_patch(plt.Circle((xys[i,2], xys[i,1]), radius=window_radius,
                            color='Red', fill=False, alpha=0.25,
                            linewidth=0.1))
    ax.text(xys[i,2]+window_radius, xys[i,1] + window_radius//2, '{}'.format(xys[i,0]), fontsize=1)

plt.imshow(data, cmap='Greys')
plt.colorbar()
plt.savefig('out/{}/led_search_areas.plot.pdf'.format(root_directory))
