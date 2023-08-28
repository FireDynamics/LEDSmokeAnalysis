import pathlib
from subprocess import call


def main(extended_logs=False):
    path = pathlib.Path(__file__).parent.absolute()
    if not extended_logs:
        call(['python', '-m', 'robot', path])
    else:
        call(['python', '-m', 'robot', '--loglevel', 'TRACE', path])


if __name__ == "__main__":
    main()
