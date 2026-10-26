*** Settings ***
Resource        global_keywords.resource

Suite Setup  Init Suite
Suite Teardown  Clean Up Suite

*** Keywords ***
Init Suite
    Remove Directory    ${WORKDIR}      recursive=True
    Create Directory    ${WORKDIR}
    Change Directory    ${WORKDIR}
    Create Test Data

Clean Up Suite
    # Windows cannot remove a directory while it is the working directory of a process,
    # so the suite leaves it before deleting it.
    Change Directory    ${EXECDIR}
    Remove Directory    ${WORKDIR}      recursive=True
