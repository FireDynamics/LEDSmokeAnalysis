import argparse


def add_parser_arguments_tools(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add parser arguments related to tools.

    :param parser: ArgumentParser object to which the arguments are added.
    :type parser: argparse.ArgumentParser
    :return: Modified ArgumentParser object with added arguments.
    :rtype: argparse.ArgumentParser
    """

    parser.add_argument('--prepare_images', action='store_true')
    return parser


def add_parser_arguments_data_extraction(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add parser arguments related to data extraction.

    :param parser: ArgumentParser object to which the arguments are added.
    :type parser: argparse.ArgumentParser
    :return: Modified ArgumentParser object with added arguments.
    :rtype: argparse.ArgumentParser
    """
    parser.add_argument('-s1', '--step_1', '--find_search_areas', action='store_true',
                        help='STEP1: analyse a reference image to find the LED positions and their labels.')
    parser.add_argument('-s2', '--step_2', '--analyse_positions', action='store_true',
                        help='STEP2: finds the LED array to which each LED belongs.')
    parser.add_argument('-s3', '--step_3', '--analyse_photo', action='store_true',
                        help='STEP3: finds the changes in light intensity')
    parser.add_argument('-s3_fast', '--step_3_fast', '--analyse_photo_fast', action='store_true',
                        help='Step 3 but without the fits.')
    parser.add_argument('-conf', '--config', nargs='*', default=None,
                        help='creates the default configuration file. optional arguments are are: img_directory, '
                             'reference_img, number_of_cores.')
    parser.add_argument('-re', '--restart', action='store_true',
                        help='Restarts step 3 of the analysis after the program was terminated before it finished.')
    parser.add_argument('-r', '--red', action='store_true',
                        help='Use the red channel for step3. Default value.')
    parser.add_argument('-g', '--green', action='store_true',
                        help='Use the green channel for step3')
    parser.add_argument('-b', '--blue', action='store_true',
                        help='Use the blue channel for step3.')
    parser.add_argument('-rgb', '--rgb', action='store_true',
                        help='Run step3 for each channel.')
    parser.add_argument('-coord', '--coordinates', action='store_true',
                        help='Calculates the 3D coordinates from the coordinates given in the configfile and the '
                             'reference image.')
    return parser


def add_parser_arguments_testing(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add parser arguments related to testing.

    :param parser: ArgumentParser object to which the arguments are added.
    :type parser: argparse.ArgumentParser
    :return: Modified ArgumentParser object with added arguments.
    :rtype: argparse.ArgumentParser
    """
    parser.add_argument('-atest', '--acceptance_test', action='store_true',
                        help='Runs the acceptance test suit')
    parser.add_argument('-atest_debug', '--acceptance_test_debug', action='store_true',
                        help='Runs acceptance test suit in debug mode')
    return parser


def add_parser_arguments_demo(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add parser arguments related to running a demo.

    :param parser: ArgumentParser object to which the arguments are added.
    :type parser: argparse.ArgumentParser
    :return: Modified ArgumentParser object with added arguments.
    :rtype: argparse.ArgumentParser
    """
    parser.add_argument('-d', '--demo', action='store_true',
                        help='Flag to indicate that the LEDSA demo should run. Must be used with --setup or --run.')
    parser.add_argument('--setup', type=str,
                        help='Path for the LEDSA demo setup. Use with --demo.')
    parser.add_argument('--run', action='store_true',
                        help='Flag to indicate that the LEDSA demo should run. Must be used with --setup.')
    parser.add_argument('--n_cores', type=int,
                        help='Flag to indicate that the LEDSA demo should run. Must be used with --run.')

    return parser


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
    parser.add_argument('-conf_a', '--config_analysis', nargs='*', default=None,
                        help='creates the analysis configuration file.')
    parser.add_argument('--cc', '--color_correction', action='store_true',
                        help='Applies color correction matrix before calculating the extinction coefficients. Use only, if'
                             'the reference property is not already color corrected.')
    parser.add_argument('--cc_channels', default=[0, 1, 2], action='extend', nargs="+", type=int,
                        help='Channels, to which color correcten gets applied. Default 0 1 2')
    return parser
