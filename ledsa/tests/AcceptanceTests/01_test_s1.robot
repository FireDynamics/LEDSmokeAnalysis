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

Step One Keeps The Help Texts Of The Config
    [Documentation]     The run fills in values and writes the config back, which must not
    ...                 cost the config the comments explaining its options.
    Create Config
    ${before} =     Count Config Help Texts
    Should Be True  ${before} > 0
    Start Step One
    ${after} =      Count Config Help Texts
    Should Be Equal As Integers     ${after}    ${before}


*** Keywords ***
Start Step One
    [Arguments]  ${use_config}=True
    Log     Starting python -m ledsa -s1
    ${leds} =   Execute Ledsa S1    ${use_config}
    RETURN  ${leds}

Hundred leds should be found and the pdf created
    [Arguments]  ${leds}
    Should Be Equal     ${leds}     100
    File Should Exist   ${WORKDIR}${/}plots${/}led_search_areas.plot.pdf
