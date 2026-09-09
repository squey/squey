###############################################################################
# Configure build system
###############################################################################
# Picks the generator before the first configuration, for a plain `cmake -S src
# -B <dir>` invocation. A build started from CMakePresets.json names its own
# generator and does not go through this.
# see https://stackoverflow.com/questions/11269833/cmake-selecting-a-generator-within-cmakelists-txt
option(USE_NINJA "Use ninja build system" ON)
if (USE_NINJA)
    find_program(NINJA_EXECUTABLE ninja)
    if (NINJA_EXECUTABLE)
        execute_process(COMMAND ${NINJA_EXECUTABLE} --version
                        OUTPUT_VARIABLE NINJA_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET)
    endif()
    if (NINJA_VERSION VERSION_GREATER_EQUAL "1.10.0")
        message(STATUS "Using ninja build system as it is available.")
        set(CMAKE_GENERATOR "Ninja" CACHE INTERNAL "" FORCE)
    else ()
        message(WARNING "Ninja build system not available, fallback to the default build system.")
        set(USE_NINJA OFF)
    endif ()
endif ()
