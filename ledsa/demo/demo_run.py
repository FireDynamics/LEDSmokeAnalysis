
import os
from ledsa.__main__ import main as ledsa_main

def run_demo(num_cores=1):
    """
    Run the demo with the provided number of cores.

    :param num_cores: Number of cores to be used in the demo.
    :type num_cores: int, optional
    """
    if 'simulation' in os.listdir():
        os.chdir('simulation')
    else:
        demo_root_dir = input("Please enter the root directory of the demo, where the 'image_data' and 'simulation' directories are located...")
        os.chdir(os.path.join(demo_root_dir, 'simulation'))

    if num_cores != 1:
        from .demo_setup import _edit_config_files
        _edit_config_files(simulation_path='.', num_cores=num_cores)

    # run s1-s3
    ledsa_main(['-s1'])
    ledsa_main(['-s2'])
    ledsa_main(['-s3_fast', '-rgb'])
    ledsa_main(['-coord'])

    # run analysis
    ledsa_main(['--analysis'])
