include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libparallelview/src/include)
include_directories(AFTER ${OPENGL_INCLUDE_DIR})
include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libpicviz/src/include)

include_directories(AFTER ${GLUT_INCLUDE_DIR})

# Qt
include(${QT_USE_FILE})

add_definitions(${PICVIZ_DEFINITIONS})
add_definitions(${OpenMP_CXX_FLAGS})

SET(SRC_FILES
PVTools.cpp
PVZoneTree.cpp
simple_lines_float_view.cpp
)

SET(HEADER_FILES
)

SET(MOC_HEADERS
)

QT4_WRAP_CPP(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})

add_library(pvparallelview SHARED ${SRC_FILES} ${HEADER_FILES} ${MOC_SRC})
set_target_properties(pvparallelview PROPERTIES SOVERSION 1)

target_link_libraries(pvparallelview ${PICVIZ_LIBRARY} ${QT_LIBRARIES} ${GLEW_LIBRARY} ${GLUT_glut_LIBRARY} ${OPENGL_LIBRARIES} ${PVKERNEL_LIBRARIES} ${TBB_LIBRARIES} ${Boost_LIBRARIES} ${OpenMP_LIBRARIES})

INSTALL(TARGETS pvparallelview LIBRARY DESTINATION .
                               RUNTIME DESTINATION .)
