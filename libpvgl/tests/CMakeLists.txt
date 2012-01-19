include_directories(AFTER ${GLEW_INCLUDE_PATH})
include_directories(AFTER ${OPENGL_INCLUDE_DIR})
include_directories(AFTER ${FREETYPE_INCLUDE_DIRS})

include_directories(AFTER ${GLUT_INCLUDE_DIR})

set(QT_USE_QTOPENGL TRUE)
# Qt
include(${QT_USE_FILE})

add_definitions(${OpenMP_CXX_FLAGS})

SET(TBB_SRC_FILES
main_tbb.cpp
Point.cpp
)

SET(SRC_FILES
main.cpp
View.cpp
Point.cpp
)

SET(HEADER_FILES
)

SET(MOC_HEADERS
View.h
)

QT4_WRAP_CPP(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})

add_definitions(-march=native -O3)
add_executable(run ${SRC_FILES} ${MOC_SRC} ${MOC_HEADERS})
add_executable(run-tbb ${TBB_SRC_FILES})

target_link_libraries(run ${QT_LIBRARIES} ${GLEW_LIBRARY} ${GLUT_glut_LIBRARY} ${OPENGL_LIBRARIES} ${FREETYPE_LIBRARY}  ${TBB_LIBRARIES} ${OpenMP_LIBRARIES})
target_link_libraries(run-tbb ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY} ${GLEW_LIBRARY} ${GLUT_glut_LIBRARY} ${OPENGL_LIBRARIES} ${FREETYPE_LIBRARY} ${TBB_LIBRARIES} ${OpenMP_LIBRARIES})
