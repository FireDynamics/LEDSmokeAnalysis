# this file is executed if the package is run via python -m ledsa [arguments]

import argparse
import sys

from .ledsa import LEDSA
from .core import ledsa_conf as lc


parser = argparse.ArgumentParser(description=
                                 'Allows the analysis of light dampening of LEDs behind a smoke screen.')
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
                    help='Use the red channel for the analysis. Default value.')
parser.add_argument('--g', '-g', '--green', action='store_true',
                    help='Use the green channel for the analysis')
parser.add_argument('--b', '-b', '--blue', action='store_true',
                    help='Use the blue channel for the analysis.')
parser.add_argument('-rgb', '--rgb', action='store_true',
                    help='Make the analysis for each channel.')
parser.add_argument('--coordinates', '-coord', action='store_true',
                    help='Calculates the 3D coordinates from the coordinates given in the configfile and the '
                         'reference image.')
parser.add_argument('--atest', '-atest', action='store_true',
                    help='Runs the acceptance test suit')
args = parser.parse_args()

print('ledsa runs with the following arguments:')
print(args)

if len(sys.argv) == 1:
    print('Please give an argument. Get help with --h')

if args.config is not None:
    if len(args.config) == 0:
        lc.ConfigData(load_config_file=False)
    if len(args.config) == 1:
        lc.ConfigData(load_config_file=False, img_directory=args.config[0])
    if len(args.config) == 2:
        lc.ConfigData(load_config_file=False, img_directory=args.config[0], reference_img=args.config[1])
    if len(args.config) == 3:
        lc.ConfigData(load_config_file=False, img_directory=args.config[0], reference_img=args.config[1],
                      multicore_processing=True, num_of_cores=args.config[2])

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
    ledsa = LEDSA(build_experiment_infos=False, channels=channels)
    if args.s1:
        ledsa.find_search_areas(ledsa.config['find_search_areas']['reference_img'])
        ledsa.plot_search_areas(ledsa.config['find_search_areas']['reference_img'])
    if args.s2:
        ledsa.match_leds_to_led_arrays()

if args.s3:
    ledsa = LEDSA(build_experiment_infos=True, channels=channels)
    ledsa.setup_step3()
    ledsa.process_image_data()

if args.s3_fast:
    ledsa = LEDSA(build_experiment_infos=True, channels=channels, fit_leds=False)
    ledsa.setup_step3()
    ledsa.process_image_data()

if args.re:
    ledsa = LEDSA(build_experiment_infos=False, channels=channels)
    ledsa.setup_restart()
    ledsa.process_image_data()

if args.coordinates:
    from ledsa.ledpositions.coordinates import calculate_coordinates
    calculate_coordinates()

# if args.atest:
#     # noinspection PyUnresolvedReferences
#     import tests.AcceptanceTests.__main__
