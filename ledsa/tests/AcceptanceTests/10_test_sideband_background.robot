*** Settings ***
Resource  global_keywords.resource

Force Tags  analysis  background_subtraction  sideband

*** Variables ***
${BINARY}            analysis${/}channel0${/}all_parameters.h5
# Every analysed quantity gets a directory of its own below the one of the solver
${EXTINCTION_DIR}    analysis${/}extinction_coefficients${/}linear
${SUM_DIR}           ${EXTINCTION_DIR}${/}sum_col_val
${BGSUB_DIR}         ${EXTINCTION_DIR}${/}bgsub_sum_col_val
${SUM_RESULT}        ${SUM_DIR}${/}extinction_coefficients_linear_channel_0_sum_col_val_led_array_0.csv
${SUM_REFERENCE}     ${SUM_DIR}${/}extinction_coefficients_linear_channel_0_sum_col_val_led_array_0_sidebg_ref.csv
${BGSUB_RESULT}      ${BGSUB_DIR}${/}extinction_coefficients_linear_channel_0_bgsub_sum_col_val_led_array_0.csv
${BGSUB_REFERENCE}   ${BGSUB_DIR}${/}extinction_coefficients_linear_channel_0_bgsub_sum_col_val_led_array_0_sidebg_ref.csv

*** Test Cases ***
Default Results Are Not Changed By The Sideband Columns
    [Documentation]    Results with the existing reference properties sum_col_val and bgsub_sum_col_val must be
    ...                identical whether the step 3 files contain the side band columns or not (data format
    ...                before the side band pedestal was added). The binary is removed before each comparison
    ...                run, because it is only created from the step 3 files if it does not exist.
    Change Directory    ${WORKDIR}
    Execute Ledsa    -s3_fast
    Remove File    ${BINARY}
    Create And Fill Config Analysis    linear
    Execute Ledsa    --analysis
    Copy File    ${SUM_RESULT}    ${SUM_REFERENCE}
    Create And Fill Config Analysis    linear    bgsub_sum_col_val
    Execute Ledsa    --analysis
    Copy File    ${BGSUB_RESULT}    ${BGSUB_REFERENCE}
    Strip Bgsub Column From Led Position Files    num_columns=6
    Remove File    ${BINARY}
    Execute Ledsa    --analysis
    ${identical} =    Check Extinction Coefficient Files Are Identical    ${BGSUB_RESULT}    ${BGSUB_REFERENCE}
    Should Be True    ${identical}
    Create And Fill Config Analysis    linear
    Execute Ledsa    --analysis
    ${identical} =    Check Extinction Coefficient Files Are Identical    ${SUM_RESULT}    ${SUM_REFERENCE}
    Should Be True    ${identical}

Sidebg Reference Property Matches Ground Truth On Clean Images
    [Documentation]    Without background the side bands contain nothing to subtract and the sidebg property
    ...                reproduces the ground truth like the default property.
    Run Sideband Analysis    sideband_clean    0
    Check Sidebg Results    2
    Check Sidebg Results    3
    Check Sidebg Results    4

Background Light Biases Sum But Not Sidebg
    [Documentation]    With a constant additive background the default property underestimates the extinction
    ...                while the sidebg property stays close to the ground truth.
    Run Sideband Analysis    sideband_background    15
    Check Background Results    2
    Check Background Results    3
    Check Background Results    4

*** Keywords ***
Run Sideband Analysis
    [Arguments]    ${dir}    ${background}
    [Documentation]    The test images are only 50 px wide, so the side bands are placed 12 to 20 px across
    ...                the strip instead of the default 25 to 60 px.
    Create Directory    ${WORKDIR}${/}${dir}
    Change Directory    ${WORKDIR}${/}${dir}
    Create Test Data    background=${background}
    Create Config
    Set Config Option    analyse_photo    sideband_distances    12 20
    Execute Ledsa    -s1
    Execute Ledsa    -s2
    Execute Ledsa    --coordinates
    Execute Ledsa    -s3_fast
    Create And Fill Config Analysis    linear
    Execute Ledsa    --analysis
    Create And Fill Config Analysis    linear    sidebg_sum_col_val
    Execute Ledsa    --analysis

Check Sidebg Results
    [Arguments]    ${image_id}
    ${rmse} =    Check Input Vs Computed Extinction Coefficients    ${image_id}    linear
    ...    reference_property=sidebg_sum_col_val
    Should Be True    ${rmse} < 0.05

Check Background Results
    [Arguments]    ${image_id}
    ${rmse_sum} =    Check Input Vs Computed Extinction Coefficients    ${image_id}    linear
    ${rmse_sidebg} =    Check Input Vs Computed Extinction Coefficients    ${image_id}    linear
    ...    reference_property=sidebg_sum_col_val
    # same slack as for bgsub in suite 08: the test data are random and the background clips the LED peaks of
    # the 8 bit images at 255; in addition the side band correction has to remove most of the bias of the sum
    Should Be True    ${rmse_sum} > 0.05
    Should Be True    ${rmse_sidebg} < 0.06
    Should Be True    ${rmse_sidebg} < 0.5 * ${rmse_sum}
