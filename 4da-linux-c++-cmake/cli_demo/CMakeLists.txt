file(
    GLOB 
    CLI_DEMO_SOURCE_FILES
    cli_demo.cpp
)

add_executable(
    cli_demo
    ${CLI_DEMO_SOURCE_FILES}
)
