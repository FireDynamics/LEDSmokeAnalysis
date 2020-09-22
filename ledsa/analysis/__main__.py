from ledsa.analysis import plot_functions as pf
from ledsa.analysis import calculations as calc
import matplotlib.pyplot as plt

#pf.create_binary_data()


figure1 = plt.figure()
# pf.plot_z_fitpar(figure, 'A', 100, 0, 1)
# pf.plot_z_fitpar(figure, 'A', 200, 0, 1)
# pf.plot_z_fitpar(figure, 'A', 810, 0, 1)
# plt.show(block=False)
# figure = plt.figure()
# pf.plot_t_fitpar(figure, 600, 'A', 0, 1, 1244, 0)
# # pf.plot_t_fitpar(figure, 0, 'A', 0, 1, 1244, 0)

# pf.plot_t_fitpar_with_moving_average(figure, 20, 'normalized_A', 0, 1, 1244)
# pf.plot_t_fitpar_with_moving_average(figure, 24, 'normalized_A', 0, 1, 1244)
# pf.plot_t_fitpar_with_moving_average(figure, 23, 'normalized_A', 0, 1, 1244)
# pf.plot_t_fitpar_with_moving_average(figure, 29, 'normalized_A', 0, 1, 1244)
pf.plot_z_fitpar(figure1, 'sum_col_val', 1, 0, (2, 3, 4))

#figure2 = plt.figure()
#pf.plot_z_fitpar_from_average(figure2, 'mean_col_val', 1, 0, (2, 3, 4), window_size=51)

plt.show()
