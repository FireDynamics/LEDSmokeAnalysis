import argparse
import os

from ledsa.analysis import ExtinctionCoefficientsNumeric as ECN
from ledsa.analysis.ConfigDataAnalysis import ConfigDataAnalysis
from ledsa.analysis.Experiment import Experiment
from ledsa.analysis.ExperimentData import ExperimentData
# from ledsa.analysis.__main__ import apply_cc_on_ref_property

from ledsa.core.ConfigData import ConfigData
from ledsa.data_extraction.DataExtractor import DataExtractor


def run_data_extraction_arguments(args: argparse.Namespace) -> None:
    """
    Execute actions based on data extraction arguments.

    :param args: Parsed command line arguments.
    :type args: argparse.Namespace
    """
    if args.config is not None: # TODO: remove additional options from config parser argument
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
        if args.red:
            channels.append(0)
        if args.green:
            channels.append(1)
        if args.blue:
            channels.append(2)
    if len(channels) == 0:
        channels.append(0)

    if args.red or args.green or args.blue or args.rgb and not args.step_3_fast:
        args.step_3 = True

    if args.step_1 or args.step_2:
        de = DataExtractor(build_experiment_infos=False, channels=channels)
        if args.step_1:
            de.find_search_areas(de.config['find_search_areas']['reference_img'])
        if args.step_2:
            de.match_leds_to_led_arrays()

    if args.step_3:
        de = DataExtractor(build_experiment_infos=True, channels=channels)
        de.setup_step3()
        de.process_image_data()

    if args.step_3_fast:
        de = DataExtractor(build_experiment_infos=True, channels=channels, fit_leds=False)
        de.setup_step3()
        de.process_image_data()

    if args.restart:
        channels = [0, 1, 2]  # TODO: just for testing
        de = DataExtractor(build_experiment_infos=False, channels=channels, fit_leds=False)
        de.setup_restart()
        de.process_image_data()

    if args.coordinates:
        from ledsa.ledpositions.coordinates import calculate_coordinates
        calculate_coordinates()


def run_testing_arguments(args) -> None:
    """
    Execute actions based on testing arguments.

    :param args: Parsed command line arguments.
    :type args: argparse.Namespace
    """

    if args.acceptance_test:
        from ledsa.tests.AcceptanceTests.__main__ import main
        main()

    if args.acceptance_test_debug:
        from ledsa.tests.AcceptanceTests.__main__ import main
        main(extended_logs=True)


def run_demo_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """
    Execute actions based on demo arguments.

    :param args: Parsed command line arguments.
    :type args: argparse.Namespace
    """
    if not args.setup and not args.run:
        parser.error("--demo requires either --setup or --run argument.")
    if args.n_cores and not args.run:
        parser.error("--n_cores requires the --run argument.")

    if args.setup:
        image_data_url = 'https://zenodo.org/record/7016689/files/image_data_V001_Cam01.zip?download=1'
        from ledsa.demo.demo_setup import setup_demo
        setup_demo(destination_path=args.setup, image_data_url=image_data_url)
    if args.run:
        from ledsa.demo.demo_run import run_demo
        if args.n_cores:
            run_demo(num_of_cores=args.n_cores)
        else:
            run_demo()

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
            out_file = os.path.join(os.getcwd(), '../analysis', 'AbsorptionCoefficients',
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
