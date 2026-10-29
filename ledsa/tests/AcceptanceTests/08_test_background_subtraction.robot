*** Settings ***
Resource  global_keywords.resource

Suite Setup  Extract Step 3 Results

Force Tags  analysis  background_subtraction

*** Variables ***
${BINARY}            analysis${/}channel0${/}all_parameters.h5
# Every analysed quantity gets a directory of its own below the one of the solver
${EXTINCTION_DIR}    analysis${/}extinction_coefficients${/}linear
${SUM_DIR}           ${EXTINCTION_DIR}${/}sum_col_val
${BGSUB_DIR}         ${EXTINCTION_DIR}${/}bgsub_sum_col_val
${SUM_RESULT}        ${SUM_DIR}${/}extinction_coefficients_linear_channel_0_sum_col_val_led_array_0.csv
${SUM_REFERENCE}     ${SUM_DIR}${/}extinction_coefficients_linear_channel_0_sum_col_val_led_array_0_before_strip.csv
${BGSUB_RESULT}      ${BGSUB_DIR}${/}extinction_coefficients_linear_channel_0_bgsub_sum_col_val_led_array_0.csv

*** Test Cases ***
Every Configured Quantity Is Analysed In One Run
    [Documentation]    config_analysis.ini can name several quantities, which one analysis run
    ...                computes one after another, each into a directory of its own instead of
    ...                overwriting the results of the quantity before it.
    Change Directory    ${WORKDIR}
    Remove Directory    ${EXTINCTION_DIR}    recursive=True
    Create And Fill Config Analysis    linear    sum_col_val bgsub_sum_col_val
    Execute Ledsa    --analysis
    File Should Exist    ${SUM_RESULT}
    File Should Exist    ${BGSUB_RESULT}
    Check Both Quantities Of One Run    2
    Check Both Quantities Of One Run    3
    Check Both Quantities Of One Run    4

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
    # the binary is only created from the step 3 files if it does not exist
    Remove File    ${BINARY}
    Execute Ledsa    --analysis
    Copy File    ${SUM_RESULT}    ${SUM_REFERENCE}
    Strip Bgsub Column From Led Position Files
    Remove File    ${BINARY}
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

Fitted Step 3 Output Can Be Analysed
    [Documentation]    Step 3 with fits writes the main and the fit columns into one row; the analysis must
    ...                read these files (both reference properties) and reproduce the ground truth.
    Create Directory    ${WORKDIR}${/}fitted
    Change Directory    ${WORKDIR}${/}fitted
    Create Test Data
    Create Config
    Execute Ledsa    -s1
    Execute Ledsa    -s2
    Execute Ledsa    --coordinates
    Execute Ledsa    -s3
    Create And Fill Config Analysis    linear
    Execute Ledsa    --analysis
    Create And Fill Config Analysis    linear    bgsub_sum_col_val
    Execute Ledsa    --analysis
    Check Fitted Results    2
    Check Fitted Results    3
    Check Fitted Results    4

*** Keywords ***
Check Both Quantities Of One Run
    [Arguments]    ${image_id}
    ${rmse_sum} =    Check Input Vs Computed Extinction Coefficients    ${image_id}    linear
    ${rmse_bgsub} =    Check Input Vs Computed Extinction Coefficients    ${image_id}    linear
    ...    reference_property=bgsub_sum_col_val
    Should Be True    ${rmse_sum} < 0.05
    Should Be True    ${rmse_bgsub} < 0.06

Extract Step 3 Results
    [Documentation]    The suites before may leave the work directory without step 3 results (suite 07 empties
    ...                analysis/channel0), so extract them here and drop a binary of an earlier run.
    Change Directory    ${WORKDIR}
    Execute Ledsa    -s3_fast
    Remove File    ${BINARY}

Check Fitted Results
    [Arguments]    ${image_id}
    ${rmse_sum} =    Check Input Vs Computed Extinction Coefficients    ${image_id}    linear
    ${rmse_bgsub} =    Check Input Vs Computed Extinction Coefficients    ${image_id}    linear
    ...    reference_property=bgsub_sum_col_val
    Should Be True    ${rmse_sum} < 0.05
    Should Be True    ${rmse_bgsub} < 0.06

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
