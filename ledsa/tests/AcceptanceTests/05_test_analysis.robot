*** Settings ***
Resource  global_keywords.resource

Force Tags  analysis_numeric

*** Test Cases ***
Step Analysis Numeric
    Create Config Analysis
    Start Step Analysis Numeric

Plot Results
    Plot Extinction Coefficients

Check Extinction Coefficient Results
    Check Results    1
    Check Results    2
    Check Results    3
    Check Results    4


*** Keywords ***
Start Step Analysis Numeric
    Log     Step Analysis Numeric
    Execute Ledsa   -s3_fast
    Execute Ledsa   --analysis

Plot Extinction Coefficients
    Log     Plot Extinction Coefficients
    Plot Input Vs Computed Extinction Coefficients

Check Results
    [Arguments]  ${image_id}
    Log     Check Results
    ${rmse} =   Check Input Vs Computed Extinction Coefficients    ${image_id}
    Rmse Should Be Small   ${rmse}

Rmse Should Be Small
    [Arguments]  ${rmse}
    Should Be True   ${rmse} < 0.02
