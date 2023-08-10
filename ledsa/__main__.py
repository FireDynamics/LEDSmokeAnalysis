# this file is executed if the package is run via python -m ledsa [arguments]

import argparse
import sys

from .data_extraction.DataExtractor import DataExtractor
from .core.ConfigData import ConfigData
from ledsa.analysis.__main__ import add_parser_argument_analysis, run_analysis_arguments_with_extinction_coefficient


def main(argv):
    parser = argparse.ArgumentParser(description=
                                     'Allows the analysis of light dampening of LEDs behind a smoke screen.')
    add_parser_arguments_data_extraction(parser)
    add_parser_argument_analysis(parser)
    add_parser_arguments_demo(parser)
    add_parser_arguments_testing(parser)

    args = parser.parse_args(argv)

    print('ledsa runs with the following arguments:')
    print(args)

    if len(argv) == 0:
        print('Please run with an argument. --help for possible arguments.')
        exit(0)

    if args.demo:
        run_demo_arguments(args)
    else:
        run_data_extraction_arguments(args)
        run_analysis_arguments_with_extinction_coefficient(args)
        run_testing_arguments(args)


def add_parser_arguments_data_extraction(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--s1', '-s1', '--find_search_areas', action='store_true',
                        help='STEP1: analyse a reference image to find the LED positions and their labels')
    parser.add_argument('--s2', '-s2', '--analyse_positions', action='store_true',
                        help='STEP2: finds the LED array to which each LED belongs')
    parser.add_argument('--s3', '-s3', '--analyse_photo', action='store_true',
                        help='STEP3: finds the changes in light intensity')
    parser.add_argument('--s3_fast', '-s3_fast', action='store_true',
                        help='Step 3 but without the fits.')
    parser.add_argument('--config', '-conf', nargs='*', default=None,
                        help='creates the default configuration file. optional arguments are are: img_directory, '
                             'reference_img, number_of_cores.')
    parser.add_argument('--re', '-re', '--restart', action='store_true',
                        help='Restarts step 3 of the analysis after the program was terminated before it finished.')
    parser.add_argument('--r', '-r', '--red', action='store_true',
                        help='Use the red channel for step3. Default value.')
    parser.add_argument('--g', '-g', '--green', action='store_true',
                        help='Use the green channel for step3')
    parser.add_argument('--b', '-b', '--blue', action='store_true',
                        help='Use the blue channel for step3.')
    parser.add_argument('-rgb', '--rgb', action='store_true',
                        help='Run step3 for each channel.')
    parser.add_argument('--coordinates', '-coord', action='store_true',
                        help='Calculates the 3D coordinates from the coordinates given in the configfile and the '
                             'reference image.')
    return parser


def add_parser_arguments_testing(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--atest', '-atest', action='store_true',
                        help='Runs the acceptance test suit')
    parser.add_argument('--atest_debug', action='store_true',
                        help='Runs acceptance test suit in debug mode')
    return parser


def add_parser_arguments_demo(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--demo', action='store_true',
                        help='Runs the LEDSA demo. Internet connection is necessary.')
    return parser


def run_data_extraction_arguments(args):
    if args.config is not None:
        if len(args.config) == 0:
            ConfigData(load_config_file=False)
        if len(args.config) == 1:
            ConfigData(load_config_file=False, img_directory=args.config[0])
        if len(args.config) == 2:
            ConfigData(load_config_file=False, img_directory=args.config[0], reference_img=args.config[1])
        if len(args.config) == 3:
            ConfigData(load_config_file=False, img_directory=args.config[0], reference_img=args.config[1],
                       num_of_cores=args.config[2])

    channels = []
    if args.rgb:
        channels = [0, 1, 2]
    else:
        if args.r:
            channels.append(0)
        if args.g:
            channels.append(1)
        if args.b:
            channels.append(2)
    if len(channels) == 0:
        channels.append(0)

    if args.r or args.g or args.b or args.rgb and not args.s3_fast:
        args.s3 = True

    if args.s1 or args.s2:
        de = DataExtractor(build_experiment_infos=False, channels=channels)
        if args.s1:
            de.find_search_areas(de.config['find_search_areas']['reference_img'])
            de.plot_search_areas(de.config['find_search_areas']['reference_img'])
        if args.s2:
            de.match_leds_to_led_arrays()

    if args.s3:
        de = DataExtractor(build_experiment_infos=True, channels=channels)
        de.setup_step3()
        de.process_image_data()

    if args.s3_fast:
        de = DataExtractor(build_experiment_infos=True, channels=channels, fit_leds=False)
        de.setup_step3()
        de.process_image_data()

    if args.re:
        channels = [0, 1, 2]  # TODO: just for testing
        de = DataExtractor(build_experiment_infos=False, channels=channels, fit_leds=False)
        de.setup_restart()
        de.process_image_data()

    if args.coordinates:
        from ledsa.data_extraction.ledpositions.coordinates import calculate_coordinates
        calculate_coordinates()


def run_testing_arguments(args):
    if args.atest:
        from tests.AcceptanceTests.__main__ import main
        main()

    if args.atest_debug:
        from tests.AcceptanceTests.__main__ import main
        main(extended_logs=True)


def run_demo_arguments(args):
    from demo.__main__ import main as demo_main
    demo_main()


if __name__ == "__main__":
    main(sys.argv[1:])
