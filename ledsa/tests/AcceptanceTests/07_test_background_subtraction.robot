*** Settings ***
Resource  global_keywords.resource

Force Tags  analysis  background_subtraction

*** Variables ***
${EXTINCTION_DIR}    analysis${/}extinction_coefficients${/}linear
${SUM_RESULT}        ${EXTINCTION_DIR}${/}extinction_coefficients_linear_channel_0_sum_col_val_led_array_0.csv
${SUM_REFERENCE}     ${EXTINCTION_DIR}${/}extinction_coefficients_linear_channel_0_sum_col_val_led_array_0_before_strip.csv

*** Test Cases ***
Bgsub Reference Property Matches Ground Truth On Clean Images
    [Documentation]    On background-free test images there is nothing to subtract, so the
    ...                opt-in bgsub property must reproduce the ground truth just like the
    ...                default property does.
    Change Directory    ${WORKDIR}
    Create And Fill Config Analysis    linear    bgsub_sum_col_val
    Execute Ledsa    --analysis
    Check Bgsub Results    2
    Check Bgsub Results    3
    Check Bgsub Results    4

Default Analysis Is Not Changed By The Bgsub Column
    [Documentation]    Results with the default reference property must be identical whether
    ...                the step 3 files contain the new bgsub column or not (old data format).
    Change Directory    ${WORKDIR}
    Create And Fill Config Analysis    linear
    Execute Ledsa    --analysis
    Copy File    ${SUM_RESULT}    ${SUM_REFERENCE}
    Strip Bgsub Column From Led Position Files
    Execute Ledsa    --analysis
    ${identical} =    Check Extinction Coefficient Files Are Identical    ${SUM_RESULT}    ${SUM_REFERENCE}
    Should Be True    ${identical}

Background Light Biases Sum But Not Bgsub
    [Documentation]    With a constant additive background in the images the default property
    ...                underestimates the extinction while the bgsub property stays close to
    ...                the ground truth.
    Create Directory    ${WORKDIR}${/}background
    Change Directory    ${WORKDIR}${/}background
    Create Test Data    background=15
    Create Config
    Execute Ledsa    -s1
    Execute Ledsa    -s2
    Execute Ledsa    --coordinates
    Execute Ledsa    -s3_fast
    Create And Fill Config Analysis    linear
    Execute Ledsa    --analysis
    Create And Fill Config Analysis    linear    bgsub_sum_col_val
    Execute Ledsa    --analysis
    Check Background Results    2
    Check Background Results    3
    Check Background Results    4

*** Keywords ***
Check Bgsub Results
    [Arguments]    ${image_id}
    ${rmse} =    Check Input Vs Computed Extinction Coefficients    ${image_id}    linear
    ...    reference_property=bgsub_sum_col_val
    # slightly more slack than the 0.05 of the default property: the border ring of the
    # synthetic JPGs contains faint LED tails and compression artefacts that get subtracted
    Should Be True    ${rmse} < 0.06

Check Background Results
    [Arguments]    ${image_id}
    ${rmse_sum} =    Check Input Vs Computed Extinction Coefficients    ${image_id}    linear
    ${rmse_bgsub} =    Check Input Vs Computed Extinction Coefficients    ${image_id}    linear
    ...    reference_property=bgsub_sum_col_val
    Should Be True    ${rmse_sum} > 0.05
    Should Be True    ${rmse_bgsub} < 0.05
