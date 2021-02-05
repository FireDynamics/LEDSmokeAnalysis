*** Settings ***
Resource  global_keywords.resource

Test Teardown  Empty Directory  ${WORKDIR}${/}analysis${/}channel0${/}

Force Tags  step_3

*** Test Cases ***
Step Three
    Start Step Three
    Result table should be generated

Step Three Wihtout Fit
    Start Step Three Without Fit
    Result table should be generated

Repeat Step Three After Interruption
    Restart Step Three
    Result table should be generated

*** Keywords ***
Start Step Three
    Log     Starting python -m ledsa -s3
    Execute Ledsa   -s3

Start Step Three Without Fit
    Log     Starting python -m ledsa -s3 -fast
    Execute Ledsa   -s3_fast

Restart Step Three
    Log     Starting python -m ledsa -re
    Execute Ledsa   -re

Result table should be generated
    Directory Should Not Be Empty   ${WORKDIR}${/}analysis${/}channel0${/}
