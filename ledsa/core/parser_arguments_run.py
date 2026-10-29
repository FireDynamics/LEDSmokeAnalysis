import argparse

from ledsa.analysis import ExtinctionCoefficientsNonLinear as ECN
from ledsa.analysis import ExtinctionCoefficientsLinear as ECA

from ledsa.analysis.ConfigDataAnalysis import ConfigDataAnalysis
from ledsa.analysis.ConfigDataStacked import ConfigDataStacked
from ledsa.analysis.Experiment import Experiment
from ledsa.analysis.ExperimentData import ExperimentData
from ledsa.analysis.StackedExtinctionCoefficients import StackedExtinctionCoefficients
from ledsa.analysis.temperature_correction import load_temperature_corrections
from ledsa.analysis.data_preparation import apply_color_correction
import numpy as np

from ledsa.core.ConfigData import ConfigData
from ledsa.core.file_handling import get_extinction_coefficients_file_path
from ledsa.data_extraction.DataExtractor import DataExtractor



def run_data_extraction_arguments(args: argparse.Namespace) -> None:
    """
    Execute actions based on data extraction arguments.

    :param args: Parsed command line arguments.
    :type args: argparse.Namespace
    """
    if args.config is not None:  # TODO: remove additional options from config parser argument
        if len(args.config) == 0:
            ConfigData(load_config_file=False)
        if len(args.config) == 1:
            ConfigData(load_config_file=False, img_directory=args.config[0])
        if len(args.config) == 2:
            ConfigData(load_config_file=False, img_directory=args.config[0], ref_img_id=args.config[1])
        if len(args.config) == 3:
            ConfigData(load_config_file=False, img_directory=args.config[0], ref_img_id=args.config[1],
                       num_cores=args.config[2])
    elif (args.step_1 or args.step_2 or args.step_3 or args.step_3_fast or args.restart or
          args.red or args.green or args.blue or args.rgb or args.coordinates):
        # If any data extraction argument is given but not config, check if config file exists
        try:
            with open('config.ini', 'r', encoding='utf-8-sig') as f:
                pass
        except FileNotFoundError:
            raise FileNotFoundError('config.ini not found in working directory! Please create it with argument "--config".')

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
            de.find_search_areas()
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
        # Use the channels determined earlier in the function
        de = DataExtractor(build_experiment_infos=False, channels=channels, fit_leds=False)
        de.setup_restart()
        de.process_image_data()

    if args.coordinates:
        from ledsa.ledpositions.coordinates import calculate_coordinates
        calculate_coordinates()


def run_testing_arguments(args: argparse.Namespace) -> None:
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
            run_demo(num_cores=args.n_cores)
        else:
            run_demo()


def apply_cc_on_ref_property(ex_data, cc_channels=None) -> None:
    """
    Apply color correction on every analysed quantity and save it in the binary as column {quantity}_cc.

    :param ex_data: Experiment data containing the reference properties
    :type ex_data: ExperimentData
    :param cc_channels: Channels to apply color correction on, defaults to None
    :type cc_channels: list, optional
    """
    try:
        cc_matrix = np.genfromtxt('mean_all_cc_matrix_integral.csv', delimiter=',')
    except FileNotFoundError:
        print('File: mean_all_cc_matrix_integral.csv containing the color correction matrix not found')
        exit(1)
    for reference_property in ex_data.reference_properties:
        apply_color_correction(cc_matrix, on=reference_property, channels=cc_channels)


def run_analysis_arguments(args) -> None:
    """
    Handle the configuration and preprocessing based on the command line arguments.

    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    """
    if args.config_analysis is not None:
        ConfigDataAnalysis(load_config_file=False)
    elif args.analysis or args.cc:
        # If any analysis argument is given but not config_analysis, check if config_analysis.ini file exists
        try:
            with open('config_analysis.ini', 'r', encoding='utf-8-sig') as f:
                pass
        except FileNotFoundError:
            raise FileNotFoundError('config_analysis.ini not found in working directory! Please create it with argument "--config_analysis".')

    if args.cc:
        ex_data = ExperimentData()
        apply_cc_on_ref_property(ex_data, args.cc_channels)

    if args.config_stacked is not None:
        ConfigDataStacked.create_template(
            filename=args.config_stacked,
            num_simulations=args.n_simulations,
        )


def run_stacked_analysis(args) -> None:
    """
    Run stacked multi-camera extinction coefficient calculation.

    Reads ``config_stacked.ini`` from the current working directory, then
    iterates over every virtual LED-array label and colour channel defined
    therein, computing and saving the joint extinction coefficients.

    :param args: Parsed command line arguments.
    :type args: argparse.Namespace
    """
    if not args.analysis_stacked:
        return

    try:
        open('config_stacked.ini', encoding='utf-8-sig').close()
    except FileNotFoundError:
        raise FileNotFoundError(
            'config_stacked.ini not found in working directory! '
            'Create a template with: python -m ledsa -conf_s'
        )

    cfg = ConfigDataStacked()

    labels = cfg.get_led_array_ids()
    channels = cfg.camera_channels
    reference_properties = cfg.reference_properties
    print(
        f'Stacked analysis: {len(labels)} LED array(s) x '
        f'{len(channels)} channel(s) x {len(reference_properties)} quantity(s), '
        f'solver={cfg.solver!r}'
    )

    for reference_property in reference_properties:
        for led_array_id in labels:
            for channel in channels:
                print(f'\n-- quantity={reference_property}  array={led_array_id}  channel={channel} --')
                sec = StackedExtinctionCoefficients(cfg, led_array_id, channel, reference_property)
                print(sec)
                if cfg.num_cores > 1:
                    print(f'Running on {cfg.num_cores} cores.')
                    sec.calc_and_set_coefficients_mp(cfg.num_cores)
                else:
                    sec.calc_and_set_coefficients()
                sec.save()


def run_analysis_arguments_with_extinction_coefficient(args) -> None:
    """
    Run the extinction coefficient calculation based on the command line arguments.

    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    """
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
    # read before the first coefficient is computed, so that an unreadable file stops the run here
    temperature_corrections = load_temperature_corrections(ex_data.reduction_files)

    # Determine which solver to use based on config
    solver = ex_data.solver.lower()

    # Warn if linear solver is used but nonlinear parameters are provided
    if solver == 'linear':
        print("Using linear solver for extinction coefficient calculation.")
        print("Note: weighting_preference and weighting_curvature parameters are ignored when using the linear solver.")
    elif solver == 'nonlinear' or solver == 'numeric':
        print("Using nonlinear solver for extinction coefficient calculation.")
        solver = 'nonlinear'  # Normalize the name
    else:
        print(f"Warning: Unknown solver type '{solver}'. Defaulting to linear solver.")
        print("Note: weighting_preference and weighting_curvature parameters are ignored when using the linear solver.")
        solver = 'linear'

    print(f'Quantities to analyse: {" ".join(ex_data.reference_properties)}')
    for reference_property in ex_data.reference_properties:
        for array in ex_data.led_arrays:
            for channel in ex_data.channels:
                print(f'\n-- quantity={reference_property}  array={array}  channel={channel} --')
                ex = Experiment(layers=ex_data.layers, led_array=array, camera=ex_data.camera, channel=channel,
                                merge_led_arrays=ex_data.merge_led_arrays)

                if solver == 'nonlinear':
                    eca = ECN.ExtinctionCoefficientsNonLinear(ex, reference_property=reference_property,
                                                              num_ref_imgs=ex_data.num_ref_images,
                                                              ref_img_indices=ex_data.ref_img_indices,
                                                              average_images=ex_data.average_images,
                                                              n_summarize=ex_data.n_summarize,
                                                              temperature_correction=temperature_corrections.get(channel),
                                                              weighting_curvature=ex_data.weighting_curvature,
                                                              weighting_preference=ex_data.weighting_preference,
                                                              num_iterations=ex_data.num_iterations)
                elif solver == 'linear':
                    eca = ECA.ExtinctionCoefficientsLinear(ex, reference_property=reference_property,
                                                           num_ref_imgs=ex_data.num_ref_images,
                                                           ref_img_indices=ex_data.ref_img_indices,
                                                           average_images=ex_data.average_images,
                                                           n_summarize=ex_data.n_summarize,
                                                           temperature_correction=temperature_corrections.get(channel),
                                                           lambda_reg=ex_data.lambda_reg,
                                                           snr_full_weight=ex_data.snr_full_weight,
                                                           min_layer_weight=ex_data.min_layer_weight,
                                                           num_moving_avg_images=ex_data.num_moving_avg_images)
                else:
                    raise ValueError(f"Invalid solver type '{solver}'. Must be 'linear' or 'nonlinear'.")
                if ex_data.n_cpus > 1:
                    print(f"Calculation of extinction coefficients runs on {ex_data.n_cpus} cpus!")
                    eca.calc_and_set_coefficients_mp(ex_data.n_cpus)
                else:
                    print("Calculation of extinction coefficients runs on a single cpu!")
                    eca.calc_and_set_coefficients()
                eca.save()
                out_file = get_extinction_coefficients_file_path(solver, channel, reference_property, array)
                print(f"{out_file} created!")
