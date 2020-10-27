*** Settings ***
Resource        global_keywords.resource

Suite Setup  Init Suit
Suite Teardown  Remove Directory    ${WORKDIR}      recursive=True

*** Keywords ***
Init Suit
    Remove Directory    ${WORKDIR}      recursive=True
    Create Directory    ${WORKDIR}
    Change Directory    ${WORKDIR}
    Create Test Image