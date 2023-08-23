from ledsa.__main__ import main as ledsa_main

def run_demo(num_of_cores=1):
    if num_of_cores != 1:
        from .demo_setup import edit_config_files
        edit_config_files(simulation_path='.', num_of_cores=num_of_cores, setup=False)
    # run s1-s3

    ledsa_main(['-s1'])
    ledsa_main(['-s2'])
    ledsa_main(['-s3_fast'])
    ledsa_main(['-coord'])

    # run analysis
    ledsa_main(['--analysis'])

    # create some interesting plots