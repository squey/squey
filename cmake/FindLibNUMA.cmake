find_path(NUMA_INCLUDE_DIR numa.h HINTS /usr/include)
find_library(NUMA_LIBRARY NAMES numa HINTS /usr/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(libnuma DEFAULT_MSG NUMA_LIBRARY NUMA_INCLUDE_DIR)

set(NUMA_LIBRARIES ${NUMA_LIBRARY})
set(NUMA_INCLUDE_DIRS ${NUMA_INCLUDE_DIR})

mark_as_advanced(NUMA_INCLUDE_DIR HWLOC_LIBRARY)
