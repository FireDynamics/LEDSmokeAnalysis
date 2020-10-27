*** Settings ***
Resource        global_keywords.resource

Suite Setup      Create And Fill Config

Force Tags  step_2


*** Test Cases ***
Test Step Two
    Start Step Two
    Pdf with lines should be created
    Line indice tabels should be created


*** Keywords ***
Start Step Two
    Log     Starting python -m ledsa -s2
    Execute Ledsa   -s2

Pdf with lines should be created
    File Should Exist   ${WORKDIR}${/}plots${/}led_lines.pdf

Line indice tabels should be created
    File Should Exist   ${WORKDIR}${/}analysis${/}line_indices_000.csv