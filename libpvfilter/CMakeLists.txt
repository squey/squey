include_directories(AFTER ${PVCORE_INCLUDE_DIRS})
include_directories(AFTER ${PVFILTER_INCLUDE_DIRS})
include_directories(AFTER ${ICU_INCLUDE})

set(QT_DONT_USE_QTGUI true)

# Set PVFILTER_NORMALIZE_DIR
if(WIN32)
        set(PVFILTER_NORMALIZE_DIR "normalize-filters")
else(WIN32)
        set(PVFILTER_NORMALIZE_DIR "${CMAKE_INSTALL_PREFIX}/share/pvfilter/normalize-filters/")
endif(WIN32)
add_definitions(-DPVFILTER_NORMALIZE_DIR="${PVFILTER_NORMALIZE_DIR}")

add_definitions(${OpenMP_CXX_FLAGS})

add_subdirectory(src)
add_subdirectory(plugins)

add_subdirectory(tests)
