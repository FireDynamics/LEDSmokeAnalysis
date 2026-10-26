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

Analysis Skips Images Without Extraction Results
    [Documentation]     An image whose pixel data could not be read keeps its ID in the image infos but
    ...                 produces no result file, so the extinction coefficients must leave it out and still
    ...                 pair every remaining image with its own experiment time.
    Corrupt Image   test_img_3.jpg  truncate
    Execute Ledsa   -s3_fast
    Corrupted Images Report Should List     test_img_3.jpg  extraction of the LED intensities
    Create And Fill Config Analysis     linear
    Execute Ledsa   --analysis
    Analysis Should Exclude Image   test_img_3.jpg
    # The left out image makes the coefficients cover fewer times than the image infos, which is no
    # mismatch of the time axes and must not be reported as one
    Extco Time Axis Should Be Accepted

*** Keywords ***
Restore Test Images
    Restore Corrupted Images
    Empty Directory     ${WORKDIR}${/}analysis${/}channel0${/}
    Remove Directory    ${WORKDIR}${/}analysis${/}extinction_coefficients${/}     recursive=True

Corrupted Images Should Be Reported
    File Should Exist   ${WORKDIR}${/}analysis${/}corrupted_images.csv

Result Table Should Be Generated
    Directory Should Not Be Empty   ${WORKDIR}${/}analysis${/}channel0${/}

Result File Should Exist For Image
    [Arguments]     ${img_name}
    ${img_id} =     Get Analysis Image Id   ${img_name}
    File Should Exist   ${WORKDIR}${/}analysis${/}channel0${/}${img_id}_led_positions.csv
