# this file is executed if the package is run via python -m ledsa [arguments]

import argparse
from ledsa import LEDSA
from ledsa import ledsa_conf as lc


parser = argparse.ArgumentParser(description=
                                 'Allows the analysis of light dampening of LEDs behind a smoke screen.')
parser.add_argument('--s1', '-s1', '--find_search_areas', action='store_true',
                    help='STEP1: analyse a reference image to find the LED positions and their labels')
parser.add_argument('--s2', '-s2', '--analyse_positions', action='store_true',
                    help='STEP2: finds the LED array to which each LED belongs')
parser.add_argument('--s3', '-s3', '--analyse_photo', action='store_true',
                    help='STEP3: finds the changes in light intensity')
parser.add_argument('--config', '-c', nargs='*', default=None,
                    help='creates the default configuration file. optional arguments are are: img_directory, '
                         'reference_img, number_of_cores.')
parser.add_argument('--re', '-re', '--restart', action='store_true',
                    help='Restarts step 3 of the analysis after the program was terminated before it finished.')
args = parser.parse_args()

print('ledsa runs with the following arguments:')
print(args)

if args.config is None and args.s1 and not args.s2 and not args.s3 and not args.re:
    args.config = []
    args.s1 = args.s2 = args.s3 = True

if args.config is not None:
    if len(args.config) == 0:
        lc.ConfigData()
    if len(args.config) == 1:
        lc.ConfigData(img_directory=args.config[0])
    if len(args.config) == 2:
        lc.ConfigData(img_directory=args.config[0], reference_img=args.config[1])
    if len(args.config) == 3:
        lc.ConfigData(img_directory=args.config[0], reference_img=args.config[1],
                      multicore_processing=True, num_of_cores=args.config[2])
if args.s1 or args.s2 or args.s3:
    ledsa = LEDSA()
    if args.s1:
        ledsa.find_search_areas(ledsa.config['find_search_areas']['reference_img'])
        ledsa.plot_search_areas(ledsa.config['find_search_areas']['reference_img'])
    if args.s2:
        ledsa.analyse_positions()
        ledsa.plot_lines()
    if args.s3:
        ledsa.process_image_data()

if args.re:
    ledsa = LEDSA()
    ledsa.find_calculated_imgs()
    ledsa.process_image_data()