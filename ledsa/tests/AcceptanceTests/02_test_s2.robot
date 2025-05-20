*** Settings ***
Resource        global_keywords.resource

Suite Setup     Create And Fill Config

Force Tags  step_2


*** Test Cases ***
Step Two
    Start Step Two
    Pdf with LED arrays should be created
    LED array indices tables should be created



*** Keywords ***
Start Step Two
    Log     Starting python -m ledsa -s2
    Execute Ledsa   -s2

Pdf with LED arrays should be created
    File Should Exist   ${WORKDIR}${/}plots${/}led_arrays.pdf

LED array indices tables should be created
    File Should Exist   ${WORKDIR}${/}analysis${/}line_indices_000.csv