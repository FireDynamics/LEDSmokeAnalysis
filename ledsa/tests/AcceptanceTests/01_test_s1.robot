*** Settings ***
Resource        global_keywords.resource

Test Teardown   Remove File         ${WORKDIR}${/}config.ini

Force Tags  step_1


*** Test Cases ***
Step One With Existing Config File
    Create Config
    ${infos} =  Start Step One
    Hundred leds should be found and the pdf created    ${infos}

Step One Without Existing Config File
    ${infos} =  Start Step One      use_config=${FALSE}
    Hundred leds should be found and the pdf created    ${infos}


*** Keywords ***
Start Step One
    [Arguments]  ${use_config}=True
    Log     Starting python -m ledsa -s1
    ${leds} =   Execute Ledsa S1    ${use_config}
    [Return]  ${leds}

Hundred leds should be found and the pdf created
    [Arguments]  ${leds}
    Should Be Equal     ${leds}     100
    File Should Exist   ${WORKDIR}${/}plots${/}led_search_areas.plot.pdf
