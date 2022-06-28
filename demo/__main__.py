def main():
    # download imgs and demo conifg file
    # run s1-s3
    from ledsa.__main__ import main as ledsa_main

    ledsa_main(['-s1'])
    ledsa_main(['-s2'])
    ledsa_main(['-s3_fast'])
    ledsa_main(['-coordinates'])

    # run analysis
    ledsa_main(['--default_input'])
    ledsa_main(['--analysis', '--cc'])

    # create some interesting plots


if __name__ == "__main__":
    main()
