###############################################################################
# Configure build system 
###############################################################################
# Use ninja build system if available
# see https://stackoverflow.com/questions/11269833/cmake-selecting-a-generator-within-cmakelists-txt
OPTION(USE_NINJA "Use ninja build system" ON)
if (USE_NINJA)
    execute_process(COMMAND ninja --version ERROR_QUIET OUTPUT_VARIABLE NINJA_VERSION)
    if ("${NINJA_VERSION}" GREATER_EQUAL "1.7.2")
        message(STATUS "Using ninja build system as it is available.")
        set(CMAKE_GENERATOR "Ninja" CACHE INTERNAL "" FORCE)
    else ()
        message(WARNING "Ninja build system not available, fallback to the default build system.")
        set(USE_NINJA OFF)
    endif ()
endif ()
