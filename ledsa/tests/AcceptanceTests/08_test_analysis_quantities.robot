*** Settings ***
Resource  global_keywords.resource

Suite Setup  Extract Step 3 Results

Force Tags  analysis

*** Variables ***
${BINARY}            analysis${/}channel0${/}all_parameters.h5
# Every analysed quantity gets a directory of its own below the one of the solver
${EXTINCTION_DIR}    analysis${/}extinction_coefficients${/}linear
${SUM_RESULT}        ${EXTINCTION_DIR}${/}sum_col_val${/}extinction_coefficients_linear_channel_0_sum_col_val_led_array_0.csv
${MEAN_RESULT}       ${EXTINCTION_DIR}${/}mean_col_val${/}extinction_coefficients_linear_channel_0_mean_col_val_led_array_0.csv

*** Test Cases ***
Every Configured Quantity Is Analysed In One Run
    [Documentation]    config_analysis.ini can name several quantities, which one analysis run
    ...                computes one after another, each into a directory of its own instead of
    ...                overwriting the results of the quantity before it. The mean of a search area
    ...                is its sum divided by its number of pixels, so it has to reproduce the ground
    ...                truth as well.
    Change Directory    ${WORKDIR}
    Remove Directory    ${EXTINCTION_DIR}    recursive=True
    Create And Fill Config Analysis    linear    sum_col_val mean_col_val
    Execute Ledsa    --analysis
    File Should Exist    ${SUM_RESULT}
    File Should Exist    ${MEAN_RESULT}
    Check Both Quantities Of One Run    2
    Check Both Quantities Of One Run    3
    Check Both Quantities Of One Run    4

Fitted Step 3 Output Can Be Analysed
    [Documentation]    Step 3 with fits writes the main and the fit columns into one row; the analysis must
    ...                read these files and reproduce the ground truth.
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
    Check Fitted Results    2
    Check Fitted Results    3
    Check Fitted Results    4

*** Keywords ***
Extract Step 3 Results
    [Documentation]    The suites before may leave the work directory without step 3 results (suite 07 empties
    ...                analysis/channel0), so extract them here and drop a binary of an earlier run.
    Change Directory    ${WORKDIR}
    Execute Ledsa    -s3_fast
    Remove File    ${BINARY}

Check Both Quantities Of One Run
    [Arguments]    ${image_id}
    ${rmse_sum} =    Check Input Vs Computed Extinction Coefficients    ${image_id}    linear
    ${rmse_mean} =    Check Input Vs Computed Extinction Coefficients    ${image_id}    linear
    ...    reference_property=mean_col_val
    Should Be True    ${rmse_sum} < 0.05
    Should Be True    ${rmse_mean} < 0.05

Check Fitted Results
    [Arguments]    ${image_id}
    ${rmse} =    Check Input Vs Computed Extinction Coefficients    ${image_id}    linear
    Should Be True    ${rmse} < 0.05
