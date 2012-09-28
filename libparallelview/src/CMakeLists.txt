#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libparallelview/src/include)
include_directories(AFTER ${OPENGL_INCLUDE_DIR})
include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libpicviz/src/include)

include_directories(AFTER ${GLUT_INCLUDE_DIR})
if (CUDA_FOUND)
	include_directories(AFTER ${CUDA_INCLUDE_DIRS})
endif(CUDA_FOUND)

set(LINK_LIBRARIES ${PICVIZ_LIBRARY} ${PVHIVE_LIBRARIES} ${QT_LIBRARIES} ${GLEW_LIBRARY} ${GLUT_glut_LIBRARY} ${OPENGL_LIBRARIES} ${PVKERNEL_LIBRARIES} ${TBB_LIBRARIES} ${Boost_LIBRARIES} ${OpenMP_LIBRARIES})

# CUDA
set(CUDA_PROPAGATE_HOST_FLAGS "OFF")
set(CUDA_NVCC_FLAGS "-Xcompiler=-fPIC -gencode arch=compute_20,code=sm_20" "-Xptxas=-v")
#set(CUDA_NVCC_FLAGS "-g -G -Xcompiler=-fPIC -gencode arch=compute_20,code=sm_20" "-Xptxas=-v")

# Qt
include(${QT_USE_FILE})

add_definitions(${PICVIZ_DEFINITIONS})
add_definitions(${OpenMP_CXX_FLAGS})

SET(SRC_FILES
PVBCIBuffers.cpp
PVBCIDrawingEngineCuda.cpp
PVParallelView.cpp
PVTools.cpp
PVZoomedZoneTree.cpp
PVZoomedParallelScene.cpp
PVZoomedParallelView.cpp
PVLinesView.cpp
PVFullParallelScene.cpp
PVFullParallelView.cpp
PVSelectionSquareGraphicsItem.cpp
PVZonesManager.cpp
PVZoneTree.cpp
PVZoneTreeBase.cpp
PVZoneTreeNoAlloc.cpp
simple_lines_float_view.cpp
simple_lines_int_view.cpp
PVAxisLabel.cpp
PVAxisGraphicsItem.cpp
PVSelectionGenerator.cpp
PVQuadTree.cpp
PVLibView.cpp
PVSlidersManager.cpp
PVSlidersGroup.cpp
PVAbstractAxisSlider.cpp
PVAbstractAxisSliders.cpp
PVAbstractRangeAxisSliders.cpp
PVSelectionAxisSlider.cpp
PVSelectionAxisSliders.cpp
PVZoomAxisSlider.cpp
PVZoomAxisSliders.cpp
)

set(CUDA_SRC_FILES
)

SET(HEADER_FILES
include/pvparallelview/common.h
include/pvparallelview/PVBCIBuffers.h
include/pvparallelview/PVBCICode.h
include/pvparallelview/PVBCode.h
include/pvparallelview/PVContainerZoneTree.h
include/pvparallelview/PVLinesView.h
include/pvparallelview/PVParallelView.h
include/pvparallelview/PVPODBlockContainer.h
include/pvparallelview/PVQuadTree.h
include/pvparallelview/PVTools.h
include/pvparallelview/PVZoneProcessing.h
include/pvparallelview/PVZonesManager.h
include/pvparallelview/PVZoneTreeBase.h
include/pvparallelview/PVZoneTree.h
include/pvparallelview/PVZoneTreeNoAlloc.h
include/pvparallelview/PVZoomedZoneTree.h
include/pvparallelview/PVZoomedParallelScene.h
include/pvparallelview/PVZoomedParallelView.h
include/pvparallelview/PVSlidersManager.h
include/pvparallelview/PVAxisLabel.h
include/pvparallelview/PVAxisGraphicsItem.h
include/pvparallelview/PVFullParallelView.h
include/pvparallelview/PVFullParallelScene.h
include/pvparallelview/PVSlidersGroup.h
include/pvparallelview/PVAbstractAxisSlider.h
include/pvparallelview/PVAbstractAxisSliders.h
include/pvparallelview/PVSelectionAxisSlider.h
include/pvparallelview/PVSelectionAxisSliders.h
include/pvparallelview/PVZoomAxisSlider.h
include/pvparallelview/PVZoomAxisSliders.h
)

SET(MOC_HEADERS
include/pvparallelview/PVRenderingJob.h
include/pvparallelview/PVFullParallelView.h
include/pvparallelview/PVFullParallelScene.h
include/pvparallelview/PVSelectionSquareGraphicsItem.h
include/pvparallelview/PVZonesManager.h
include/pvparallelview/PVZoomedParallelScene.h
include/pvparallelview/PVSlidersGroup.h
include/pvparallelview/PVAxisLabel.h
include/pvparallelview/PVAbstractAxisSlider.h
include/pvparallelview/PVAbstractAxisSliders.h
include/pvparallelview/PVSelectionAxisSliders.h
include/pvparallelview/PVZoomAxisSliders.h
)

QT4_WRAP_CPP(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})

if (CUDA_FOUND)
	cuda_compile(CUDA_KERNELS cuda/bci_cuda_z24.cu)
	set(SRC_FILES ${SRC_FILES} ${CUDA_SRC_FILES} ${CUDA_KERNELS})
	set(LINK_LIBRARIES ${LINK_LIBRARIES} ${CUDA_LIBRARIES} /opt/cuda/lib64/libnpp.so)
endif (CUDA_FOUND)

add_library(pvparallelview SHARED ${SRC_FILES} ${HEADER_FILES} ${MOC_SRC})
set_target_properties(pvparallelview PROPERTIES SOVERSION 1)

target_link_libraries(pvparallelview ${LINK_LIBRARIES})

INSTALL(TARGETS pvparallelview LIBRARY DESTINATION .
                               RUNTIME DESTINATION .)
