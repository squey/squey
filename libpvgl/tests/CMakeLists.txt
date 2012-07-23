#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libpvgl/src/include)
include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libpicviz/src/include)
include_directories(AFTER ${GLEW_INCLUDE_PATH})
include_directories(AFTER ${OPENGL_INCLUDE_DIR})
include_directories(AFTER ${FREETYPE_INCLUDE_DIRS})
include_directories(AFTER ${GLUT_INCLUDE_DIR})
include_directories(AFTER include/)

remove_definitions(-DPVGL_SHARE_DIR=${PVGL_SHARE_DIR})
set(QT_USE_QTOPENGL TRUE)
# Qt
include(${QT_USE_FILE})

add_definitions(${OpenMP_CXX_FLAGS})

#add_definitions(-march=native -O3 -ffast-math -g)
set(COMMON_LIBS ${QT_LIBRARIES} ${GLEW_LIBRARY} ${GLUT_glut_LIBRARY} ${OPENGL_LIBRARIES} ${FREETYPE_LIBRARY} ${TBB_LIBRARIES} ${OpenMP_LIBRARIES} ${PVKERNEL_LIBRARIES} ${PICVIZ_LIBRARIES} ${PVSDK_LIBRARIES} ${PVGL_LIBRARIES})
# Tests programs

add_executable(T_process_file_view process_file_view.cpp)
target_link_libraries(T_process_file_view ${COMMON_LIBS})

add_executable(T_ad2g_widget ad2g_widget.cpp)
target_link_libraries(T_ad2g_widget ${COMMON_LIBS})

add_executable(Tprofile_pvview profile_pvview.cpp)
target_link_libraries(Tprofile_pvview ${COMMON_LIBS})
