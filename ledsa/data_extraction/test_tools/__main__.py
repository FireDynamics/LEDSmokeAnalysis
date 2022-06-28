from .log_analysis import FitAnalyser
import sys

if len(sys.argv) == 1:
    from .plot_coordinates import plot_coordinates
    plot_coordinates()

elif len(sys.argv) == 2:
    from .plot_coordinates import compare_coordinates
    compare_coordinates()
else:
    arguments = ''
    for argument in sys.argv[1:]:
        arguments += argument + ' '
    fa = FitAnalyser(arguments)
    fa.plot_image()

    while True:
        answer = input('Do you want to refit the image (y/n)? ')
        if answer == 'y':
            fa.refit_image()
            fa.plot_image()
        elif answer == 'n':
            break
        else:
            print('Please answer only with y for yes or n for no.\n')


# TEST FOR PROJECTION ON THE PLANE
# from ..core._led_helper import _fit_plane
# from ..core._led_helper import _project_points_to_plane
# from ..core._led_helper import _get_plane_coordinates
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# points = np.ndarray((3, 6))
# points[0] = (1, 3, 7, 5, 3, 4)
# points[1] = (5, 4.5, 5, 5.2, 5.3, 4.8)
# points[2] = (54, 21, 4, 68, 12, 1)
#
# plane = _fit_plane(points)
# proj_points = _project_points_to_plane(points, plane)
#
# coord = _get_plane_coordinates(proj_points, plane)
# print(coord)
#
# zz, xx = np.meshgrid(range(70), range(10))
# y = (-plane[0] * xx - plane[2] * zz - plane[3]) * 1. /plane[1]
#
# plot3d = plt.figure().gca(projection='3d')
# plot3d.plot_surface(xx, y, zz, alpha=0.2)
#
# plot3d.scatter(points[0], points[1], points[2])
# plot3d.scatter(proj_points[0], proj_points[1], proj_points[2])
#
# plt.show()
