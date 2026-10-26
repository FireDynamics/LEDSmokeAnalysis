*** Settings ***
Resource  global_keywords.resource

Force Tags  analysis  stacked

*** Variables ***
# The test with the skipped image needs its own data, because the data extraction steps cannot be
# repeated in a directory they already ran in.
${SKIPPEDDIR}   ${WORKDIR}${/}stacked_skipped_image

*** Test Cases ***
Step Stacked Analysis Linear Solver
    Create Stacked Test Data
    Setup Camera Simulation    cam0    0.5
    Setup Camera Simulation    cam1    2.5
    Change Directory    ${WORKDIR}${/}stacked
    Create And Fill Config Stacked    cam0    cam1
    Execute Ledsa    -as
    Plot Stacked Extinction Coefficients    linear
    Check Stacked Extinction Coefficient Results    linear

Stacked Analysis Skips Images Without Extraction Results
    [Documentation]     An image whose pixel data could not be read keeps its ID in the image infos of
    ...                 its camera but produces no LED intensities. Its time must therefore drop out of
    ...                 the common time axis, while every remaining time keeps the images of its own.
    Create Stacked Test Data    ${SKIPPEDDIR}
    Setup Camera Simulation    cam0    0.5    root=${SKIPPEDDIR}
    Setup Camera Simulation    cam1    2.5    root=${SKIPPEDDIR}    corrupted_img=test_img_3.jpg
    Corrupted Images Report Should List     test_img_3.jpg  extraction of the LED intensities
    Change Directory    ${SKIPPEDDIR}${/}stacked
    # The test images lie 1.25 s apart, so the tolerance has to stay below that. A larger one would
    # match the time of the skipped image to one of its neighbours instead of dropping it.
    Create And Fill Config Stacked    cam0    cam1    time_sync_tolerance=0.5
    Execute Ledsa    -as
    Stacked Analysis Should Exclude Image   cam1    test_img_3.jpg
    Check Stacked Results At Image Time     cam1    2    linear
    Check Stacked Results At Image Time     cam1    4    linear


*** Keywords ***
Create Stacked Test Data
    [Arguments]    ${root}=${WORKDIR}
    Log    Creating stacked test data for two cameras at different heights
    Create Directory    ${root}
    Change Directory    ${root}
    Create Test Data Stacked

Setup Camera Simulation
    [Arguments]    ${cam_dir}    ${cam_z}    ${root}=${WORKDIR}    ${corrupted_img}=${EMPTY}
    Log    Setting up camera simulation in ${cam_dir} with camera height z=${cam_z}
    Change Directory    ${root}${/}stacked${/}${cam_dir}
    Create And Fill Config For Stacked Cam    ${cam_z}
    Execute Ledsa    -s1
    Execute Ledsa    -s2
    Execute Ledsa    --coordinates
    # The image is broken before the intensities are extracted, so that it is skipped without ever
    # leaving a result file behind
    Run Keyword If    '${corrupted_img}' != '${EMPTY}'    Corrupt Image    ${corrupted_img}    truncate
    Execute Ledsa    -s3_fast

Plot Stacked Extinction Coefficients
    [Arguments]    ${solver}
    Log    Plotting stacked extinction coefficients with ${solver} solver
    Change Directory    ${WORKDIR}${/}stacked
    Plot Stacked Input Vs Computed Extinction Coefficients    ${solver}

Check Stacked Extinction Coefficient Results
    [Arguments]    ${solver}
    Log    Checking stacked extinction coefficient results for ${solver} solver
    Check Stacked Results    2    ${solver}
    Check Stacked Results    3    ${solver}
    Check Stacked Results    4    ${solver}

Check Stacked Results
    [Arguments]    ${image_id}    ${solver}
    Log    Checking stacked results for image ${image_id}
    ${rmse} =    Check Stacked Input Vs Computed    ${image_id}    ${solver}
    Stacked Rmse Should Be Small    ${rmse}

Check Stacked Results At Image Time
    [Arguments]    ${sim_dir}    ${image_id}    ${solver}
    Log    Checking stacked results at the experiment time of image ${image_id}
    ${rmse} =    Check Stacked Input Vs Computed At Image Time    ${sim_dir}    ${image_id}    ${solver}
    Stacked Rmse Should Be Small    ${rmse}

Stacked Rmse Should Be Small
    [Arguments]    ${rmse}
    Should Be True    ${rmse} < 0.05
