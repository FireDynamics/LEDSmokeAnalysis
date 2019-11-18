from .log_analysis import FitAnalyser
import sys

arguments = ''
for argument in sys.argv[1:]:
    arguments += argument + ' '
fa = FitAnalyser(arguments)
fa.plot_image()

