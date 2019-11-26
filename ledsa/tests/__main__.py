from .log_analysis import FitAnalyser
import sys

if len(sys.argv) == 1:
    from .plot_coordinates import plot_coordinates
    plot_coordinates()

else:
    arguments = ''
    for argument in sys.argv[1:]:
        arguments += argument + ' '
    fa = FitAnalyser(arguments)
    fa.plot_image()

    while True:
        answer = input('Do you want to refit the image (y/n)? ')
        if answer == 'y':
            fa.refit_image()
            fa.plot_image()
        elif answer == 'n':
            break
        else:
            print('Please answer only with y for yes or n for no.\n')
