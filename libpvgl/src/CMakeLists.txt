include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libpvgl/src/include)
include_directories(AFTER ${GLEW_INCLUDE_PATH})
include_directories(AFTER ${OPENGL_INCLUDE_DIR})
include_directories(AFTER ${FREETYPE_INCLUDE_DIRS})
include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libpicviz/src/include)

include_directories(AFTER ${CURL_INCLUDE_DIR})

include_directories(AFTER ${GLUT_INCLUDE_DIR})

# Qt
include(${QT_USE_FILE})

add_definitions(${PICVIZ_DEFINITIONS})

SET(SRC_FILES
PVAxes.cpp
PVBox.cpp
#PVCom.cpp
PVContainer.cpp
PVDrawable.cpp
PVEventLine.cpp
PVFontManager.cpp
PVGLThread.cpp
PVHBox.cpp
PVIdleManager.cpp
PVLabel.cpp
PVLayout.cpp
PVLines.cpp
PVMain.cpp
PVMap.cpp
PVMisc.cpp
views/PVParallel.cpp
views/PVScatter.cpp
views/PVScatterPlotSelectionSquare.cpp
PVSelectionSquare.cpp
PVUtils.cpp
PVVBox.cpp
PVWidget.cpp
PVWidgetManager.cpp
)

SET(WTK_SRC_FILES
windowtoolkits/core/WtkWindow.cpp
windowtoolkits/core/WtkWindowsList.cpp

windowtoolkits/freeglut3/Buffers.cpp
windowtoolkits/freeglut3/Callbacks.cpp
windowtoolkits/freeglut3/Init.cpp
windowtoolkits/freeglut3/Time.cpp
windowtoolkits/freeglut3/Window.cpp

windowtoolkits/qt/Buffers.cpp
windowtoolkits/qt/Callbacks.cpp
windowtoolkits/qt/Init.cpp
windowtoolkits/qt/Time.cpp
windowtoolkits/qt/Window.cpp
windowtoolkits/qt/global.cpp
windowtoolkits/qt/PVGLWidget.cpp
)

SET(HEADER_FILES
include/pvgl/general.h
include/pvgl/PVAxes.h
include/pvgl/PVBox.h
#include/pvgl/PVCom.h
include/pvgl/PVContainer.h
include/pvgl/PVDrawable.h
include/pvgl/PVEventLine.h
include/pvgl/PVFonts.h
include/pvgl/PVGLThread.h
include/pvgl/PVHBox.h
include/pvgl/PVIdleManager.h
include/pvgl/PVInputDevicesMapping.h
include/pvgl/PVLabel.h
include/pvgl/PVLayout.h
include/pvgl/PVLines.h
include/pvgl/PVMain.h
include/pvgl/PVMap.h
include/pvgl/PVMisc.h
include/pvgl/views/PVParallel.h
include/pvgl/views/PVScatter.h
include/pvgl/views/PVScatterPlotSelectionSquare.h
include/pvgl/PVSelectionSquare.h
include/pvgl/PVUtils.h
include/pvgl/PVVBox.h
include/pvgl/PVWidget.h
include/pvgl/PVWidgetManager.h

include/pvgl/PVConfig.h
)

SET(WTK_HEADER_FILES
windowtoolkits/core/include/WtkWindow.h
windowtoolkits/core/include/WtkWindowsList.h

windowtoolkits/core/include/Buffers.h
windowtoolkits/core/include/Init.h
windowtoolkits/core/include/Time.h
windowtoolkits/core/include/Window.h

windowtoolkits/qt/include/common.h
windowtoolkits/qt/include/global.h
windowtoolkits/qt/include/PVGLWidget.h
)

SET(MOC_HEADERS
windowtoolkits/qt/include/PVGLWidget.h
)

QT4_WRAP_CPP(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})

add_library(pvgl SHARED ${SRC_FILES} ${HEADER_FILES} ${WTK_SRC_FILES} ${WTK_HEADER_FILES} ${MOC_SRC})
set_target_properties(pvgl PROPERTIES SOVERSION 1)

target_link_libraries(pvgl ${PICVIZ_LIBRARY} ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY} ${GLEW_LIBRARY} ${GLUT_glut_LIBRARY} ${OPENGL_LIBRARIES} ${FREETYPE_LIBRARY} ${PVKERNEL_LIBRARIES} ${TBB_LIBRARIES} ${Boost_LIBRARIES} ${PVSDK_LIBRARY})

INSTALL(TARGETS pvgl LIBRARY DESTINATION .
                     RUNTIME DESTINATION .)
