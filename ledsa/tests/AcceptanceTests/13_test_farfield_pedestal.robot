*** Settings ***
Resource  global_keywords.resource

Suite Setup  Prepare Images With A Pedestal
Suite Teardown  Change Directory  ${WORKDIR}

Force Tags  farfield_pedestal


*** Variables ***
${PEDESTAL_DIR}         ${WORKDIR}${/}farfield_pedestal
# Light that does not dim with the LEDs, added to every pixel of the images. It is about a quarter
# of the sum over the search area of an undimmed LED.
${PEDESTAL}             3
# The strip runs down the middle of the images, so that bands 150 to 300 px away from it fit on
# both sides.
${IMG_WIDTH}            650
${FARFIELD_DISTANCES}   150 300


*** Test Cases ***
Step 3 Measures The Pedestal In The Far Field
    [Documentation]    With farfield_distances in config.ini, step 3 writes the pedestal of every search area,
    ...                estimated in bands beside the strip, and the sum minus it. Per pixel of the search
    ...                area it is the pixel value added to the images.
    Farfield Pedestal Per Pixel Should Be    1    ${PEDESTAL}    0.1
    Farfield Pedestal Per Pixel Should Be    4    ${PEDESTAL}    0.1

The Pedestal Corrected Sum Reproduces The Extinction Coefficients
    [Documentation]    The analysis of farbg_sum_col_val finds the extinction coefficients the images were
    ...                made with.
    Create And Fill Config Analysis    linear    farbg_sum_col_val
    Execute Ledsa    --analysis
    Extinction Coefficients Should Match The Input    2
    Extinction Coefficients Should Match The Input    3
    Extinction Coefficients Should Match The Input    4

The Plain Sum Underestimates The Extinction Coefficients
    [Documentation]    The pedestal does not dim with the LEDs, so the plain sum of the same images gives too
    ...                high a transmission and too low extinction coefficients. This is what the far-field
    ...                pedestal corrects.
    Create And Fill Config Analysis    linear    sum_col_val
    Execute Ledsa    --analysis
    ${ratio} =    Mean Ratio Of Computed To Input Extinction Coefficients    2    linear
    Should Be True    ${ratio} < 0.85


*** Keywords ***
Prepare Images With A Pedestal
    Create Directory    ${PEDESTAL_DIR}
    Change Directory    ${PEDESTAL_DIR}
    Create Test Data    img_width=${IMG_WIDTH}    pedestal=${PEDESTAL}
    Create And Fill Config    farfield_distances=${FARFIELD_DISTANCES}
    ${leds} =    Execute Ledsa S1    ${TRUE}
    Should Be Equal    ${leds}    100
    Execute Ledsa    -s2
    Execute Ledsa    --coordinates
    Execute Ledsa    -s3_fast

Extinction Coefficients Should Match The Input
    [Arguments]    ${image_id}
    ${rmse} =    Check Input Vs Computed Extinction Coefficients    ${image_id}    linear
    ...    reference_property=farbg_sum_col_val
    Should Be True    ${rmse} < 0.05
