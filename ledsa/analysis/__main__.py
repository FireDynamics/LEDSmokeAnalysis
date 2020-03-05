from ledsa.analysis import plot_functions as pf
import matplotlib.pyplot as plt

figure = plt.figure()
pf.plot_z_fitpar(figure, 'A', 'IMG_8058.JPG', 0, 0)
pf.plot_z_fitpar(figure, 'A', 'IMG_8058.JPG', 0, 3)
plt.show()
