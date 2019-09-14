import numpy as np
import matplotlib.pyplot as plt

import led_helper as led

root_directory = '2018.11/V1_C1'
img_filename = 'IMG_7460.JPG'
pos_filename = 'out/{}/led_search_areas.csv'.format(root_directory)
led_positions = np.loadtxt(pos_filename)

nled = led_positions.shape[0]

indices = led_positions[:,0]
xs = led_positions[:,2]
ys = led_positions[:,1]

xmin = np.min(xs)
xmax = np.max(xs)
ymin = np.min(ys)
ymax = np.max(ys)

img_data = led.read_file('data/{}/{}'.format(root_directory,img_filename))

fig = plt.figure(dpi=900)

plt.imshow(img_data, cmap='Greys')

ignore_indices = np.loadtxt('out/{}/ignore_indices.csv'.format(root_directory))
line_edge_indices = np.loadtxt('out/{}/line_indices.csv'.format(root_directory), delimiter=',')

line_distances = np.zeros((nled, len(line_edge_indices)))

for il in range(len(line_edge_indices)):
    i1 = int(line_edge_indices[il][0])
    i2 = int(line_edge_indices[il][1])

    print(i1, i2)

    p1x = xs[i1]
    p1y = ys[i1]
    p2x = xs[i2]
    p2y = ys[i2]

    pd = np.sqrt((p1x - p2x) ** 2 + (p1y - p2y) ** 2)
    d = np.abs(((p2y - p1y) * xs - (p2x - p1x) * ys
                + p2x * p1y - p2y * p1x) / pd)

    line_distances[:,il] = d

line_indices = []
for il in line_edge_indices:
    line_indices.append([])

for iled in range(nled):
    print(iled)

    if iled in ignore_indices: continue

    while True:
        il = np.argmin(line_distances[iled,:])
        i1 = int(line_edge_indices[il][0])
        i2 = int(line_edge_indices[il][1])

        p1x = xs[i1]
        p1y = ys[i1]
        p2x = xs[i2]
        p2y = ys[i2]

        cx = xs[iled]
        cy = ys[iled]

        d1 = np.sqrt((p1x-cx)**2 + (p1y-cy)**2)
        d2 = np.sqrt((p2x-cx)**2 + (p2y-cy)**2)
        d12 = np.sqrt((p1x-p2x)**2 + (p1y-p2y)**2) + 1e-8

        print(d1, d2, d12)

        if d1 < d12 and d2 < d12: break

        line_distances[iled, il] *= 2

    line_indices[il].append(iled)


for i in range(len(line_indices)):
    plt.scatter(xs[line_indices[i]], ys[line_indices[i]], s=0.1, label='led strip {}'.format(i))

plt.legend()
plt.savefig('out/{}/led_lines.pdf'.format(root_directory))

for i in range(len(line_indices)):
    out_file = open('out/{}/line_indices_{:03}.csv'.format(root_directory, i), 'w')
    for iled in line_indices[i]:
        out_file.write('{}\n'.format(iled))
    out_file.close()
