*** Settings ***
Resource  global_keywords.resource

Force Tags  analysis_numeric

*** Test Cases ***
Config File Exists
    Create Config Analysis
    Start Step Analysis Numeric
    Plot Extinction Coefficients

#    ${infos} =  Start Step One
#    Three leds should be found and the pdf created    ${infos}



*** Keywords ***
Start Step Analysis Numeric
    Log     Step Analysis Numeric
    Execute Ledsa   -s3_fast
    Execute Ledsa   --analysis

Plot Extinction Coefficients
    Log     Plot Extinction Coefficients
    Plot Input Vs Computed Extinction Coefficients




#Cooordinates should be calculated
#    File Should Exist   ${WORKDIR}${/}analysis${/}led_search_areas_with_coordinates.csv
