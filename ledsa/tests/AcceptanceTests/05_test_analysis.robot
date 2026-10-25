*** Settings ***
Resource  global_keywords.resource

Force Tags  analysis

*** Test Cases ***
Step Analysis Linear Solver
    Create Config Analysis    linear
    Start Step Analysis
    Plot Extinction Coefficients    linear
    Check Extinction Coefficient Results    linear

Step Analysis Nonlinear Solver
    Create Config Analysis    nonlinear
    Start Step Analysis
    Plot Extinction Coefficients    nonlinear
    Check Extinction Coefficient Results    nonlinear

Extinction Coefficients Of An Earlier Run Are Reported
    [Documentation]     The extinction coefficient files carry their own time axis, while the LED
    ...                 parameters are put on the times read from the image infos. A file left over
    ...                 from a run with other image times must not pass as part of the current one.
    Create Config Analysis    linear
    Start Step Analysis
    Extco Time Axis Should Be Accepted
    Extco Time Axis Mismatch Should Be Reported


*** Keywords ***
Start Step Analysis
    Log     Step Analysis
    Execute Ledsa   -s3_fast
    Execute Ledsa   --analysis

Plot Extinction Coefficients
    [Arguments]  ${solver}
    Log     Plot Extinction Coefficients with ${solver} solver
    Plot Input Vs Computed Extinction Coefficients    ${solver}

Check Extinction Coefficient Results
    [Arguments]  ${solver}
    Log     Check Results with ${solver} solver
    Check Results    1    ${solver}
    Check Results    2    ${solver}
    Check Results    3    ${solver}
    Check Results    4    ${solver}

Check Results
    [Arguments]  ${image_id}  ${solver}
    Log     Check Results for image ${image_id} with ${solver} solver
    ${rmse} =   Check Input Vs Computed Extinction Coefficients    ${image_id}  ${solver}
    Rmse Should Be Small   ${rmse}

Rmse Should Be Small
    [Arguments]  ${rmse}
    Should Be True   ${rmse} < 0.05
