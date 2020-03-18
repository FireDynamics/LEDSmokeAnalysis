from ledsa.analysis import plot_functions as pf
import matplotlib.pyplot as plt

figure = plt.figure()
pf.plot_z_fitpar(figure, 'A', 1, 0, 0)
pf.plot_z_fitpar(figure, 'A', 1, 0, 3)
plt.show()
figure = plt.figure()
pf.plot_t_fitpar(figure, 5, 'A', 0, 1, 3, 0)
plt.show()
