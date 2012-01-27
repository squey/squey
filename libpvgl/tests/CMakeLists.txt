include_directories(AFTER ${GLEW_INCLUDE_PATH})
include_directories(AFTER ${OPENGL_INCLUDE_DIR})
include_directories(AFTER ${FREETYPE_INCLUDE_DIRS})
include_directories(AFTER ${GLUT_INCLUDE_DIR})
include_directories(AFTER ${CUDA_INCLUDE_DIRS})
include_directories(AFTER include/)

set(QT_USE_QTOPENGL TRUE)
# Qt
include(${QT_USE_FILE})

add_definitions(${OpenMP_CXX_FLAGS})

# "Libs" files
file(GLOB CORETBB_SRC_FILES core-tbb/*.cpp)
file(GLOB GL_SRC_FILES gl/*.cpp)
file(GLOB COMMON_SRC_FILES common/*.cpp)

set(CUDA_SRC_FILES cuda/common.cpp)
set(CUDA_PROPAGATE_HOST_FLAGS "OFF")
set(CUDA_NVCC_FLAGS "-O3" "-arch=sm_20")


SET(GL_MOC_HEADERS
include/gl/view.h
)
QT4_WRAP_CPP(GL_MOC_SRC ${GL_MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})
set(GL_SRC_FILES ${GL_SRC_FILES} ${GL_MOC_SRC})

add_definitions(-march=native -O3)
set(COMMON_LIBS ${QT_LIBRARIES} ${GLEW_LIBRARY} ${GLUT_glut_LIBRARY} ${OPENGL_LIBRARIES} ${FREETYPE_LIBRARY} ${TBB_LIBRARIES} ${OpenMP_LIBRARIES})
# Tests programs

####
# TBB reduction tests
SET(TBB_SRC_FILES
${COMMON_SRC_FILES}
${CORETBB_SRC_FILES}
main_tbb.cpp
)
add_executable(run-tbb ${TBB_SRC_FILES})
target_link_libraries(run-tbb ${COMMON_LIBS} -ltbbmalloc)
####

####
# GL lines drawing
SET(SRC_FILES
${COMMON_SRC_FILES}
${CORETBB_SRC_FILES}
${GL_SRC_FILES}
main_lines.cpp
)
add_executable(run-gl ${SRC_FILES} ${MOC_SRC} ${MOC_HEADERS})
target_link_libraries(run-gl ${COMMON_LIBS})
####

####
# CUDA reduction tests
cuda_compile(CUDA_RED_KERNELS cuda/gpu_collision.cu)
set(CUDA_RED_SRC_FILES
${COMMON_SRC_FILES}
${CUDA_SRC_FILES}
${CUDA_RED_KERNELS}
main_cuda.cpp
)
add_executable(cuda_red ${CUDA_RED_SRC_FILES})
target_link_libraries(cuda_red ${COMMON_LIBS} ${CUDA_LIBRARIES})
####
