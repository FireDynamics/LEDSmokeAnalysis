*** Settings ***
Resource  global_keywords.resource

Suite Setup  Prepare Images Of A Camera Behind A Pane
Suite Teardown  Change Directory  ${WORKDIR}

Force Tags  camera_pane


*** Variables ***
${PANE_DIR}       ${WORKDIR}${/}camera_behind_a_pane
# The camera stands at y = 0 and the LEDs at y = 4, so a pane at y = 2 leaves half of every line
# of sight in the smoke.
${PANE_POINT}     0 2 0
${PANE_NORMAL}    0 1 0


*** Test Cases ***
The Light Paths Of A Camera Behind A Pane Start At The Pane
    [Documentation]    The camera looks through a window pane into the smoke, so the part of every line of
    ...                sight between the camera and the pane is free of smoke. The test images have smoke
    ...                behind the pane only. With the pane in config_analysis.ini, the extinction coefficients
    ...                match the input.
    Create And Fill Config Analysis    linear    pane_point=${PANE_POINT}    pane_normal=${PANE_NORMAL}
    Execute Ledsa    --analysis
    Extinction Coefficients Should Match The Input    2
    Extinction Coefficients Should Match The Input    3
    Extinction Coefficients Should Match The Input    4

Without The Pane The Smoke Is Spread Over The Whole Line Of Sight
    [Documentation]    Analysed as if the smoke reached the camera, which stays the default without pane_point
    ...                and pane_normal, the same images give coefficients half as large for the constant
    ...                extinction of image 2, because every light path is taken twice as long as its part in
    ...                the smoke. This is what the pane corrects.
    Create And Fill Config Analysis    linear
    Execute Ledsa    --analysis
    ${ratio} =    Mean Ratio Of Computed To Input Extinction Coefficients    2    linear
    Should Be True    0.4 < ${ratio} < 0.6


*** Keywords ***
Prepare Images Of A Camera Behind A Pane
    Create Directory    ${PANE_DIR}
    Change Directory    ${PANE_DIR}
    Create Test Data    pane_point=${PANE_POINT}    pane_normal=${PANE_NORMAL}
    Create Config
    ${leds} =    Execute Ledsa S1    ${TRUE}
    Should Be Equal    ${leds}    100
    Execute Ledsa    -s2
    Execute Ledsa    --coordinates
    Execute Ledsa    -s3_fast

Extinction Coefficients Should Match The Input
    [Arguments]    ${image_id}
    ${rmse} =    Check Input Vs Computed Extinction Coefficients    ${image_id}    linear
    Should Be True    ${rmse} < 0.05
