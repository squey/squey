include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpvgl2/src/include)
include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpvgl2/src/wtk/include)
include_directories(AFTER ${GLEW_INCLUDE_PATH})
include_directories(AFTER ${OPENGL_INCLUDE_DIR})
include_directories(AFTER ${FREETYPE_INCLUDE_DIRS})
include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpicviz/src/include)

include_directories(AFTER ${CURL_INCLUDE_DIR})

include_directories(AFTER ${GLUT_INCLUDE_DIR})

# Qt
include(${QT_USE_FILE})

add_definitions(${PICVIZ_DEFINITIONS})

SET(SRC_FILES
PVGLThread.cpp
PVMain.cpp
PVUtils.cpp
)

SET(WTK_SRC_FILES
wtk/WtkInit.cpp
)

SET(HEADER_FILES
include/pvgl/general.h
include/pvgl/PVMain.h
include/pvgl/PVUtils.h
include/pvgl/PVGLThread.h
)

SET(WTK_HEADER_FILES
wtk/include/WtkInit.h
)

add_library(pvgl SHARED ${SRC_FILES} ${HEADER_FILES} ${WTK_SRC_FILES} ${WTK_HEADER_FILES})
set_target_properties(pvgl PROPERTIES SOVERSION 1)

target_link_libraries(pvgl ${PICVIZ_LIBRARY} ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY} ${GLEW_LIBRARY} ${GLUT_glut_LIBRARY} ${OPENGL_LIBRARIES} ${FREETYPE_LIBRARY} ${PVKERNEL_LIBRARIES} ${TBB_LIBRARIES} ${Boost_LIBRARIES} ${PVSDK_LIBRARY})

INSTALL(TARGETS pvgl DESTINATION .)
