# this file is executed if the package is run via python -m ledsa [arguments]
import argparse
import sys
from typing import List

from ledsa.core.parser_arguments_declaration import (add_parser_arguments_tools, add_parser_arguments_data_extraction,
    add_parser_arguments_testing, add_parser_arguments_demo, add_parser_argument_analysis)
from ledsa.core.parser_arguments_run import run_tools_arguments, run_data_extraction_arguments, run_testing_arguments, run_demo_arguments, \
    run_analysis_arguments, run_analysis_arguments_with_extinction_coefficient


def main(argv: List[str]) -> None:
    """
    Main function to execute the LEDSA package with command line arguments.

    :param argv: Command line arguments.
    :type argv: list[str]
    """
    parser = argparse.ArgumentParser(description=
                                     'Allows the analysis of light dampening of LEDs behind a smoke screen.')
    add_parser_arguments_tools(parser)
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
        run_demo_arguments(args, parser)
    else:
        run_tools_arguments(args)
        run_data_extraction_arguments(args)
        run_analysis_arguments(args)
        run_analysis_arguments_with_extinction_coefficient(args)
        run_testing_arguments(args)



if __name__ == "__main__":
    main(sys.argv[1:])
