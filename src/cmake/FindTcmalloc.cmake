# - Find Tcmalloc
# Find the native Tcmalloc includes and library
#
#  Tcmalloc_INCLUDE_DIR - where to find Tcmalloc.h, etc.
#  Tcmalloc_LIBRARIES   - List of libraries when using Tcmalloc.
#  Tcmalloc_FOUND       - True if Tcmalloc found.

message(STATUS "CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")

find_path(Tcmalloc_INCLUDE_DIR gperftools/tcmalloc.h
    /app/inlude
    /include
    /usr/include
    /opt/local/include
    /usr/local/include
)

if (USE_TCMALLOC)
    set(Tcmalloc_NAMES tcmalloc)
else ()
    set(Tcmalloc_NAMES tcmalloc_minimal tcmalloc)
endif ()

find_library(Tcmalloc_LIBRARY
    NAMES ${Tcmalloc_NAMES}
    PATHS /app/lib /lib /usr/lib /usr/local/lib /opt/local/lib
)

message(STATUS "Tcmalloc_LIBRARY=${Tcmalloc_LIBRARY}")
message(STATUS "Tcmalloc_INCLUDE_DIR=${Tcmalloc_INCLUDE_DIR}")

if (Tcmalloc_INCLUDE_DIR AND Tcmalloc_LIBRARY)
    set(Tcmalloc_FOUND TRUE)
    set( Tcmalloc_LIBRARIES ${Tcmalloc_LIBRARY} )
else ()
    set(Tcmalloc_FOUND FALSE)
    set( Tcmalloc_LIBRARIES )
endif ()

if (Tcmalloc_FOUND)
    message(STATUS "Found Tcmalloc: ${Tcmalloc_LIBRARY}")
else ()
    message(STATUS "Not Found Tcmalloc: ${Tcmalloc_LIBRARY}")
    if (Tcmalloc_FIND_REQUIRED)
        message(STATUS "Looked for Tcmalloc libraries named ${Tcmalloc_NAMES}.")
        message(FATAL_ERROR "Could NOT find Tcmalloc library")
    endif ()
endif ()

mark_as_advanced(
    Tcmalloc_LIBRARY
    Tcmalloc_INCLUDE_DIR
)
