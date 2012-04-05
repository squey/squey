include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libparallelview/src/include)
add_definitions(${PICVIZ_DEFINITIONS})
include_directories(AFTER ${PICVIZ_INCLUDE_DIRS})

add_definitions(${OpenMP_CXX_FLAGS})
set(QT_USE_QTOPENGL TRUE)
include(${QT_USE_FILE})

set(LINK_LIBRARIES ${PICVIZ_LIBRARY} ${PVKERNEL_LIBRARY} ${QT_LIBRARIES} ${TBB_LIBRARIES} ${ICU_LIBRARY} pvparallelview ${OpenMP_LIBRARIES})

enable_testing()

add_executable(create_tree create_tree.cpp)
target_link_libraries(create_tree ${LINK_LIBRARIES})

add_executable(show_codes_gl show_codes_gl.cpp)
target_link_libraries(show_codes_gl ${LINK_LIBRARIES})
