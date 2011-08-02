include (CTest)
set(CTEST_PROJECT_NAME "libPVRush")

set(QT_USE_QTXML TRUE)
include(${QT_USE_FILE})

# set(CMAKE_BUILD_TYPE "DEBUG")

include_directories(AFTER ${PVRUSH_INCLUDE_DIRS})
include_directories(AFTER ${PVCORE_INCLUDE_DIRS})
include_directories(AFTER ${PVFILTER_INCLUDE_DIRS})
include_directories(AFTER ${ICU_INCLUDE})

set(QT_DONT_USE_QTGUI true)

##########################################
# begin: SET VARIOUS PLUGINS DIRECTORIES #
##########################################
# Set PVRUSH_NORMALIZE_DIR
if(WIN32)
        set(PVRUSH_NORMALIZE_DIR "normalize")
else(WIN32)
        set(PVRUSH_NORMALIZE_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/normalize/")
endif(WIN32)
add_definitions(-DPVRUSH_NORMALIZE_DIR="${PVRUSH_NORMALIZE_DIR}")

# Set PVRUSH_NORMALIZE_HELPERS_DIR
if(WIN32)
        set(PVRUSH_NORMALIZE_HELPERS_DIR "normalize-helpers")
else(WIN32)
        set(PVRUSH_NORMALIZE_HELPERS_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/normalize-helpers/")
endif(WIN32)
add_definitions(-DPVRUSH_NORMALIZE_HELPERS_DIR="${PVRUSH_NORMALIZE_HELPERS_DIR}")

# Set PVRUSH_DECODERS_DIR
if(WIN32)
        set(PVRUSH_DECODERS_DIR "decoders")
else(WIN32)
        set(PVRUSH_DECODERS_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/decoders/")
endif(WIN32)
add_definitions(-DPVRUSH_DECODERS_DIR="${PVRUSH_DECODERS_DIR}")

# Set PVRUSH_CACHE_DIR
if(WIN32)
        set(PVRUSH_CACHE_DIR "cache")
else(WIN32)
        set(PVRUSH_CACHE_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/cache/")
endif(WIN32)
add_definitions(-DPVRUSH_CACHE_DIR="${PVRUSH_CACHE_DIR}")

# Set PVRUSH_INPUTTYPE_DIR
if(WIN32)
	set(PVRUSH_INPUTTYPE_DIR "input-types")
else(WIN32)
	set(PVRUSH_INPUTTYPE_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/input-types/")
endif(WIN32)
add_definitions(-DPVRUSH_INPUTTYPE_DIR="${PVRUSH_INPUTTYPE_DIR}")

# Set PVRUSH_SOURCE_DIR
if(WIN32)
	set(PVRUSH_SOURCE_DIR "sources")
else(WIN32)
	set(PVRUSH_SOURCE_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/sources/")
endif(WIN32)
add_definitions(-DPVRUSH_SOURCE_DIR="${PVRUSH_SOURCE_DIR}")

########################################
# END: SET VARIOUS PLUGINS DIRECTORIES #
########################################
add_definitions(${OpenMP_CXX_FLAGS})

if(NOT WIN32)
	add_definitions(-DLIBPVRUSH -Wall -Wextra -DDEBUG)
endif(NOT WIN32)

add_subdirectory(src)
add_subdirectory(plugins)
add_subdirectory(utils)

add_subdirectory(bindings)

add_subdirectory(tests)

