from typing import List

import argparse
import os
import sys

import ledsa.analysis.ExtinctionCoefficientsNumeric as ECN
from ledsa.analysis.Experiment import Experiment
from ledsa.analysis.ExperimentData import ExperimentData
from ledsa.analysis.ConfigDataAnalysis import ConfigDataAnalysis


def main(argv: List[str]) -> None:
    """
    Main function to execute the LEDSA package with analysis related command line arguments.

    :param argv: Command line arguments.
    :type argv: list[str]
    """
    parser = argparse.ArgumentParser(description=
                                     'Calculation of the extinction coefficients.')
    parser = add_parser_argument_analysis(parser)

    args = parser.parse_args(argv)

    run_analysis_arguments(args)

    # create files with the extinction coefficients
    extionction_coefficient_calculation(args)


def add_parser_argument_analysis(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add command line arguments related to analysis to the parser.

    :param parser: ArgumentParser object
    :type parser: argparse.ArgumentParser
    :return: Modified ArgumentParser object with added arguments.
    :rtype: argparse.ArgumentParser
    """
    parser.add_argument('-a', '--analysis', action='store_true',
                        help='Activate extinction coefficient calculation if not run directly from analysis package')
    parser.add_argument('-conf_a', '--config_analysis' , nargs='*', default=None,
                        help='creates the analysis configuration file.')
    parser.add_argument('--cc', '--color_correction', action='store_true',
                        help='Applies color correction matrix before calculating the extinction coefficients. Use only, if'
                             'the reference property is not already color corrected.')
    parser.add_argument('--cc_channels', default=[0, 1, 2], action='extend', nargs="+", type=int,
                        help='Channels, to which color correcten gets applied. Default 0 1 2')
    return parser


def run_analysis_arguments(args) -> None:
    """
    Handle the configuration and preprocessing based on the command line arguments.

    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    """
    if args.config_analysis is not None:
        ConfigDataAnalysis(load_config_file=False)

    if args.cc:
        ex_data = ExperimentData()
        apply_cc_on_ref_property(ex_data)


def run_analysis_arguments_with_extinction_coefficient(args) -> None:
    """
    Run the extinction coefficient calculation based on the command line arguments.

    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    """
    run_analysis_arguments(args)
    if args.analysis:
        extionction_coefficient_calculation(args)


def extionction_coefficient_calculation(args) -> None:
    """
    Calculate extinction coefficients and save the results to a file.

    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    """
    ex_data = ExperimentData()
    ex_data.request_config_parameters()
    for array in ex_data.led_arrays:
        for channel in ex_data.channels:
            out_file = os.path.join(os.getcwd(), 'analysis', 'AbsorptionCoefficients',
                                    f'absorption_coefs_numeric_channel_{channel}_{ex_data.reference_property}_led_array_{array}.csv')
            if not os.path.exists(out_file):
                ex = Experiment(layers=ex_data.layers, led_array=array, camera=ex_data.camera, channel=channel,
                                merge_led_arrays=ex_data.merge_led_arrays)
                eca = ECN.ExtinctionCoefficientsNumeric(ex, reference_property=ex_data.reference_property,
                                                        num_ref_imgs=ex_data.num_ref_images,
                                                        weighting_curvature=ex_data.weighting_curvature,
                                                        weighting_preference=ex_data.weighting_preference,
                                                        num_iterations=ex_data.num_iterations)
                if ex_data.n_cpus > 1:
                    print(f"Calculation of extinction coefficients runs on {ex_data.n_cpus} cpus!")
                    eca.calc_and_set_coefficients_mp(ex_data.n_cpus)
                else:
                    print("Calculation of extinction coefficients runs on a single cpu!")
                    eca.calc_and_set_coefficients()
                eca.save()
                print(f"{out_file} created!")
            else:
                print(f"{out_file} already exists!")


def apply_cc_on_ref_property(ex_data) -> None:
    """
    Apply color correction on the reference property and save it in the binary as column {ref_property}_cc.

    :param ex_data: Experiment data containing the reference property
    :type ex_data: ExperimentData
    """
    import numpy as np
    from ledsa.analysis.data_preparation import apply_color_correction
    try:
        cc_matrix = np.genfromtxt('mean_all_cc_matrix_integral.csv', delimiter=',')
    except FileNotFoundError:
        print('File: mean_all_cc_matrix_integral.csv containing the color correction matrix not found')
        exit(1)
    apply_color_correction(cc_matrix, on=ex_data.reference_property, channels=args.cc_channels)


if __name__ == "__main__":
    args = sys.argv
    main(sys.argv[1:])
