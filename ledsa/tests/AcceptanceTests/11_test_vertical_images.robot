*** Settings ***
Resource  global_keywords.resource

Suite Teardown  Change Directory  ${WORKDIR}

Force Tags  vertical_images


*** Test Cases ***
Photos Taken With The Camera On Its Side Are Analysed Upright
    [Documentation]    A camera held on its side writes the pixels in the orientation of its sensor and notes
    ...                how it was held in the EXIF metadata, so a vertical photo is stored as a horizontal
    ...                image. LEDSA has to turn it upright, otherwise the LED array runs across the image
    ...                instead of along its height.
    Prepare Sideways Images    camera_on_its_side
    Run Workflow On Sideways Images

Photos Without An Orientation Entry Are Turned By The Config
    [Documentation]    Images that were exported without their EXIF entries carry no orientation, so the
    ...                rotation has to be taken from img_rotation instead.
    Prepare Sideways Images    exported_without_metadata    write_orientation=${FALSE}
    Set Config Option    DEFAULT    img_rotation    90
    Run Workflow On Sideways Images


*** Keywords ***
Prepare Sideways Images
    [Arguments]    ${dir}    ${write_orientation}=${TRUE}
    Create Directory    ${WORKDIR}${/}${dir}
    Change Directory    ${WORKDIR}${/}${dir}
    Create Test Data
    Turn Test Images Sideways    write_orientation=${write_orientation}
    Create Config

Run Workflow On Sideways Images
    ${leds} =    Execute Ledsa S1    ${TRUE}
    Should Be Equal    ${leds}    100
    Led Array Should Run Along The Image Height
    Execute Ledsa    -s2
    Execute Ledsa    --coordinates
    Execute Ledsa    -s3_fast
    Create And Fill Config Analysis    linear
    Execute Ledsa    --analysis
    Extinction Coefficients Should Match The Input    2
    Extinction Coefficients Should Match The Input    3
    Extinction Coefficients Should Match The Input    4

Led Array Should Run Along The Image Height
    [Documentation]    The test images show one LED array along the height of the image. Read sideways, the
    ...                LED positions span the width of the image instead, so they no longer describe the LED
    ...                array as it was photographed.
    ${rows}    ${columns} =    Get Led Search Area Extent
    Should Be True    ${rows} > 10 * ${columns}

Extinction Coefficients Should Match The Input
    [Arguments]    ${image_id}
    ${rmse} =    Check Input Vs Computed Extinction Coefficients    ${image_id}    linear
    Should Be True    ${rmse} < 0.05
