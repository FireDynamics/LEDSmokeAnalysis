from subprocess import call
import pathlib


path = pathlib.Path(__file__).parent.absolute()
call(['python', '-m', 'robot', path])
