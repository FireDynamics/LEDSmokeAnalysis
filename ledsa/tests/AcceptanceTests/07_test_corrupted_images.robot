*** Settings ***
Resource  global_keywords.resource

Suite Setup  Change Directory  ${WORKDIR}
Test Teardown  Restore Test Images

Force Tags  corrupted_images

*** Test Cases ***
Corrupted Images Are Skipped And Reported
    Corrupt Image   test_img_2.jpg  truncate
    Corrupt Image   test_img_3.jpg  garbage
    Execute Ledsa   -s3_fast
    Corrupted Images Should Be Reported
    # test_img_3.jpg cannot be read at all, so it is left out of the analysis while the image metadata is read
    Corrupted Images Report Should List     test_img_3.jpg  reading of the image metadata
    Analysed Image Names Should Be  test_img_1.jpg  test_img_2.jpg  test_img_4.jpg
    # test_img_2.jpg still has readable metadata, so it is skipped when its pixel data is read
    Corrupted Images Report Should List     test_img_2.jpg  extraction of the LED intensities
    Result Table Should Be Generated

Readable Images Are Analysed Despite Corrupted Ones
    Corrupt Image   test_img_3.jpg  garbage
    Execute Ledsa   -s3_fast
    Result File Should Exist For Image      test_img_1.jpg
    Result File Should Exist For Image      test_img_2.jpg
    Result File Should Exist For Image      test_img_4.jpg

*** Keywords ***
Restore Test Images
    Restore Corrupted Images
    Empty Directory     ${WORKDIR}${/}analysis${/}channel0${/}

Corrupted Images Should Be Reported
    File Should Exist   ${WORKDIR}${/}analysis${/}corrupted_images.csv

Result Table Should Be Generated
    Directory Should Not Be Empty   ${WORKDIR}${/}analysis${/}channel0${/}

Result File Should Exist For Image
    [Arguments]     ${img_name}
    ${img_id} =     Get Analysis Image Id   ${img_name}
    File Should Exist   ${WORKDIR}${/}analysis${/}channel0${/}${img_id}_led_positions.csv
