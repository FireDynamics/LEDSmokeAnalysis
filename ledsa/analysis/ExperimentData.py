import ast
from dataclasses import dataclass
from typing import List

from ledsa.analysis.Experiment import Camera, Layers
# Todo: Import init_function for request of missing data
from ledsa.analysis.ConfigDataAnalysis import ConfigDataAnalysis

@dataclass
class ExperimentData:
    """
    Dataclass containing the data for the extinction coefficient calculation from the experiment_data.txt input file
    """

    def __init__(self, load_config_file=True):
        self.config = ConfigDataAnalysis(load_config_file=load_config_file)
        self.camera = None
        self.layers = None
        self.channels = None
        self.arrays = None
        self.n_cpus = None
        self.weighting_preference = None
        self.weighting_curvature = None
        self.num_iterations = None
        self.num_ref_images = None
        self.load_experiment_data() # Todo: Does that belong here?

    def load_experiment_data(self):
        config = self.config
        num_layers = int(config['boundary_conditions']['num_of_layers'])
        domain_bounds = config.get_list_of_values('boundary_conditions', 'domain_bounds', dtype=float)
        self.channels = config.get_list_of_values('DEFAULT', 'camera_channels')
        self.arrays = config.get_list_of_values('DEFAULT', 'camera_channels')
        camera_position = config.get_list_of_values('boundary_conditions', 'camera_position')

        self.layers = Layers(num_layers, *domain_bounds)
        self.camera = Camera(*camera_position)
        self.n_cpus = int(config['DEFAULT']['num_of_cores'])

    def load_fit_parameters(self):
        config = self.config
        self.num_ref_images =int(config['DEFAULT']['num_ref_images'])
        self.weighting_preference = int(config['DEFAULT']['weighting_preference'])
        self.weighting_curvature = int(config['DEFAULT']['weighting_preference'])
        self.num_iterations = int(config['DEFAULT']['num_iterations'])


