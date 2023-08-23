*** Settings ***
Library     OperatingSystem
Library     LedsaATestLibrary.py


*** Variables ***
#${WORKDIR}     ${TEMPDIR}${/}ledsa
${WORKDIR}    /Users/kristianboerger/repos/PhD/LEDSmokeAnalysis/test_output_1
*** Keywords ***
Change Directory
    [Arguments]     ${dir}
    Change Dir      ${dir}
    Log             Changed directory to ${dir}

Create Config
    Log     Creating config.ini
    Create And Fill Config

Create Config Analysis
    Log     Creating config_analysis.ini
    Create And Fill Config Analysis

