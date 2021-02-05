*** Settings ***
Resource        global_keywords.resource

Test Teardown   Remove File         ${WORKDIR}${/}config.ini

Force Tags  step_1


*** Test Cases ***
Config File Exists
    Create Config
    ${infos} =  Start Step One
    Three leds should be found and the pdf created    ${infos}

No Config File Exists
    ${infos} =  Start Step One      use_config=${FALSE}
    Three leds should be found and the pdf created    ${infos}


*** Keywords ***
Start Step One
    [Arguments]  ${use_config}=True
    Log     Starting python -m ledsa -s1
    ${leds} =   Execute Ledsa S1    ${use_config}
    [Return]  ${leds}

Three leds should be found and the pdf created
    [Arguments]  ${leds}
    Should Be Equal     ${leds}     3
    File Should Exist   ${WORKDIR}${/}plots${/}led_search_areas.plot.pdf
