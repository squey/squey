include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${PICVIZ_INCLUDE_DIRS})
include_directories(AFTER ${PVSDK_INCLUDE_DIRS})
include_directories(AFTER ${ICU_INCLUDE})

set(QT_USE_QTOPENGL 1)

add_definitions(${COMPILER_OPTIMIZE_FLAGS})

add_subdirectory(src)
