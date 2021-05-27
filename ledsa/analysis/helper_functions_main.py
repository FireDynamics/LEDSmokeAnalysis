from dataclasses import dataclass
from typing import List
import ast

from ledsa.analysis.Experiment import Camera, Layers

@dataclass
class experiment_data:
    '''
    Dataclass containing the data for the extinction coefficient calculation from the experiment_data.txt input file
    '''
    camera: Camera
    layers: Layers
    channels: List[int]
    arrays: List[int]
    n_cpus: int

    def __str__(self):
        out = repr(self.camera) + '\n'
        out += repr(self.layers) + '\n'
        out += 'Channels: ' + repr(self.channels) + '\n'
        out += 'Arrays: ' + repr(self.arrays) + '\n'
        out += 'Number Cpus: ' + repr(self.n_cpus)
        return out

    @classmethod
    def from_str(cls, init_string: str):
        '''
        Alternative constructor using the classes string representation
        '''
        init_list = init_string.split('\n')
        return cls(eval(init_list[0], {'Camera': Camera, '__builtins__': None}),
                   eval(init_list[1], {'Layers': Layers, '__builtins__': None}),
                   ast.literal_eval(init_list[2].split(': ')[1]),
                   ast.literal_eval(init_list[3].split(': ')[1]),
                   int(init_list[4].split(': ')[1]))


def create_default_experiment_data() -> None:
    '''
    Creates the default experiment_data.txt input file for the experiment information.
    '''
    cam = Camera(0, 0, 0)
    layers = Layers(20, 0, 3)
    ex_data = experiment_data(cam, layers, [0,1,2], [0], 4)

    out_file = open('experiment_data.txt', 'w')
    out_file.write(str(ex_data))
    out_file.close()


def load_experiment_data() -> experiment_data:
    '''
    Loads experiment_data.txt.
    '''
    try:
        in_file = open('experiment_data.txt', 'r')
    except FileNotFoundError:
        print('experiment_data.txt was not found.')
        exit(1)
    except PermissionError:
        print('Missing permissions to read from experiment_data.txt')
        exit(1)
    except Exception as e:
        print('Exception while reading experiment_data.txt: ', e)
        exit(1)
    data = in_file.read()
    return experiment_data.from_str(data)