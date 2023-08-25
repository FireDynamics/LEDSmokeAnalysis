*** Settings ***
Resource  global_keywords.resource

Force Tags  coord

*** Test Cases ***
Step Coord
    Start Step Coords

*** Keywords ***
Start Step Coords
    Log     Step Coord
    Execute Ledsa   -coord

Cooordinates should be calculated
    File Should Exist   ${WORKDIR}${/}analysis${/}led_search_areas_with_coordinates.csv
