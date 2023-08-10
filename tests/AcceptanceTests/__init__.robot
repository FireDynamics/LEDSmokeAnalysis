*** Settings ***
Resource        global_keywords.resource

Suite Setup  Init Suite
Suite Teardown  Remove Directory    ${WORKDIR}      recursive=True

*** Keywords ***
Init Suite
    Remove Directory    ${WORKDIR}      recursive=True
    Create Directory    ${WORKDIR}
    Change Directory    ${WORKDIR}
    Create Test Data