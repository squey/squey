include(CTest)
set(CTEST_PROJECT_NAME "PVKernel")

set(QT_USE_QTXML TRUE)
include(${QT_USE_FILE})

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${ICU_INCLUDE})
include_directories(AFTER ${ARCHIVE_INCLUDE_DIRS})
include_directories(AFTER ${PCAP_INCLUDE_DIRS})

##########################################
# begin: SET VARIOUS PLUGINS DIRECTORIES #
##########################################

# Set PVFILTER_NORMALIZE_DIR
if(WIN32)
        set(PVFILTER_NORMALIZE_DIR "normalize-filters")
else(WIN32)
        set(PVFILTER_NORMALIZE_DIR "${CMAKE_INSTALL_PREFIX}/share/pvfilter/normalize-filters/")
endif(WIN32)
add_definitions(-DPVFILTER_NORMALIZE_DIR="${PVFILTER_NORMALIZE_DIR}")

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
add_definitions(${COMPILER_OPTIMIZE_FLAGS})

add_subdirectory(src)
add_subdirectory(plugins)

set(PVKERNEL_PLUGINS_LIST ${PLUGINS_LIST})

add_subdirectory(bindings)
#add_subdirectory(utils)
add_subdirectory(tests)
