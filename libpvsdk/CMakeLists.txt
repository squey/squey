include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${PICVIZ_INCLUDE_DIRS})
include_directories(AFTER ${ICU_INCLUDE})

add_definitions(${COMPILER_OPTIMIZE_FLAGS})

add_subdirectory(src)
