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

Analysis Reads The Column Names From The First Existing Result File
    [Documentation]     The column names of the analysis are read from one result file. An image that could
    ...                 not be read has none at all, so an unreadable first image must not keep the analysis
    ...                 from reading the names from the first file that does exist.
    Corrupt Image   test_img_1.jpg  truncate
    Execute Ledsa   -s3_fast
    Corrupted Images Report Should List     test_img_1.jpg  extraction of the LED intensities
    # image 1 is the default reference image and has no results here, so image 2 provides the reference
    Create And Fill Config Analysis     linear      num_ref_images=2
    ${stderr} =     Execute Ledsa Expecting Warning      --analysis
    # the reference window of two images still names image 1, whose absence is reported
    Should Contain      ${stderr}   reference images [1]
    Analysis Should Exclude Image   test_img_1.jpg
    Extco Time Axis Should Be Accepted

Analysis Without Any Reference Image Results Stops With An Error
    [Documentation]     Image 1 is the only reference image of the default configuration. If it could not
    ...                 be read, no reference intensities can be calculated, which has to be reported
    ...                 instead of leaving the reference empty and normalising by nothing.
    Corrupt Image   test_img_1.jpg  truncate
    Execute Ledsa   -s3_fast
    Corrupted Images Report Should List     test_img_1.jpg  extraction of the LED intensities
    Create And Fill Config Analysis     linear
    ${stderr} =     Execute Ledsa Expecting Failure      --analysis
    Should Contain      ${stderr}   hold no intensity values
    Should Contain      ${stderr}   num_ref_images or ref_img_indices

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
