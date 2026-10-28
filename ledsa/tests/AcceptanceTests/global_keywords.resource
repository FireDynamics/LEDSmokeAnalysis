*** Settings ***
Library     OperatingSystem
Library     LedsaATestLibrary.py


*** Variables ***
${WORKDIR}     ${TEMPDIR}${/}ledsa

*** Keywords ***
Change Directory
    [Arguments]     ${dir}
    Change Dir      ${dir}
    Log             Changed directory to ${dir}

Create Config
    Log     Creating config.ini
    Create And Fill Config

Create Config Analysis
    [Arguments]  ${solver}  ${average_images}=False
    Log     Creating config_analysis.ini with ${solver} solver
    Create And Fill Config Analysis    ${solver}    average_images=${average_images}
