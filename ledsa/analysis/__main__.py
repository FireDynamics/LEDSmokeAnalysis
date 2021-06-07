# from ledsa.analysis import ui
import argparse
import os

from ledsa.analysis.Experiment import Experiment
import ledsa.analysis.ExtinctionCoefficientsNumeric as ECN
from ledsa.analysis.helper_functions_main import create_default_experiment_data, load_experiment_data

parser = argparse.ArgumentParser(description=
                                 'Calculation of the extinction coefficients.')
parser.add_argument('--default_input', action='store_true',
                    help='Create an input file with a default experiemnt configuration.')
parser.add_argument('--cc', '--color_correction', action='store_true',
                    help='Applies color correction matrix before calculating the extinction coefficients. Use only, if'
                         'the reference property is not already color corrected.')
parser.add_argument('--no_mp', action='store_true',
                    help='Deactivates multi core processing.')
parser.add_argument('--ref_property', default='sum_col_val',
                    help='Changes the reference property from the default sum_col_val to the specified one.')

args = parser.parse_args()

if args.default_input:
    create_default_experiment_data()
    exit(0)

# create and use the color corrected reference property
if args.cc:
    import numpy as np
    from ledsa.analysis.calculations import apply_color_correction
    try:
        cc_matrix = np.genfromtxt('mean_all_cc_matrix_integral.csv', delimiter=',')
    except(FileNotFoundError):
        print('File: mean_all_cc_matrix_integral.csv containing the color correction matrix not found')
        exit(1)
    apply_color_correction(cc_matrix, on=args.ref_property)
    args.ref_property += '_cc'

# create files with the extinction coefficients
ex_data = load_experiment_data()
for array in ex_data.arrays:
    for channel in ex_data.channels:
        out_file = os.path.join(os.getcwd(), 'analysis', 'AbsorptionCoefficients',
                                f'absorption_coefs_numeric_channel_{channel}_{args.ref_property}_led_array_{array}.csv')
        if not os.path.exists(out_file):
            ex = Experiment(ex_data.layers, led_array=array, camera=ex_data.camera, channel=channel)
            eca = ECN.ExtinctionCoefficientsNumeric(ex, reference_property=args.ref_property)
            if args.no_mp:
                eca.calc_and_set_coefficients()
            else:
                eca.calc_and_set_coefficients_mp(ex_data.n_cpus)
            print(f"{out_file} created!")
        else:
            print(f"{out_file} already exists!")


# app = ui.GUI()
# app.mainloop()
