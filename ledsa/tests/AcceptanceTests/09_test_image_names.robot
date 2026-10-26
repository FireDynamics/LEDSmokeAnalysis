*** Settings ***
Resource  global_keywords.resource

Force Tags  image_names

*** Test Cases ***
Zero Padded Image IDs With A Format Spec
    [Documentation]    Most cameras write zero padded image IDs (DSC_0001.CR3), for which the config
    ...                documents a format spec such as {:04d}. The whole workflow must accept it, including
    ...                the start time, which is read from the first image when it is not set.
    Prepare Padded Image IDs    padded_ids
    Run Workflow And Check Results

Time Reference Image With A Zero Padded ID
    [Documentation]    The camera clock offset can be computed from a time reference image given by its ID,
    ...                which is formatted with the same template. The first test image shows 12:00:00.
    Prepare Padded Image IDs    padded_time_reference
    Set Config Option    DEFAULT    exif_time_infront_real_time    None
    Set Config Option    DEFAULT    time_ref_img_id    1
    Set Config Option    DEFAULT    time_ref_img_time    12:00:00
    Run Workflow And Check Results

*** Keywords ***
Prepare Padded Image IDs
    [Arguments]    ${dir}
    Create Directory    ${WORKDIR}${/}${dir}
    Change Directory    ${WORKDIR}${/}${dir}
    Create Test Data
    Rename Test Images To Padded IDs
    Create Config
    Set Config Option    DEFAULT    img_name_string    test_img_{:04d}.jpg

Run Workflow And Check Results
    Execute Ledsa    -s1
    Execute Ledsa    -s2
    Execute Ledsa    --coordinates
    Execute Ledsa    -s3_fast
    Analysed Image Names Should Be    test_img_0001.jpg    test_img_0002.jpg    test_img_0003.jpg    test_img_0004.jpg
    Create And Fill Config Analysis    linear
    Execute Ledsa    --analysis
    Check Padded Results    2
    Check Padded Results    3
    Check Padded Results    4

Check Padded Results
    [Arguments]    ${image_id}
    ${rmse} =    Check Input Vs Computed Extinction Coefficients    ${image_id}    linear
    Should Be True    ${rmse} < 0.05
