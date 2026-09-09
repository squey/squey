#
# @file
#

################################################
# Ressource declaration
################################################

set(SRC_FILES
PVAbstractAxisSlider.cpp
PVAbstractAxisSliders.cpp
PVAbstractRangeAxisSliders.cpp
PVAxisGraphicsItem.cpp
PVAxisLabel.cpp
PVAxisHeader.cpp
PVBCIBuffers.cpp
PVBCIBackendImageQPainter.cpp
PVBCIDrawingBackendQPainter.cpp
PVFullParallelScene.cpp
PVFullParallelView.cpp
PVFullParallelViewParamsWidget.cpp
PVFullParallelViewSelectionRectangle.cpp
PVHitGraphBlocksManager.cpp
PVHitGraphBuffer.cpp
PVHitGraphDataInterface.cpp
PVHitGraphDataOMP.cpp
PVHitGraphDataOMP_kernels_baseline.cpp
PVHitGraphDataOMP_kernels_avx2.cpp
PVHitGraphSSEHelpers.cpp
PVViewRenderingContext.cpp
PVLinesView.cpp
PVParallelView.cpp
PVRenderingPipeline.cpp
PVSelectionAxisSlider.cpp
PVSelectionAxisSliders.cpp
PVSelectionGenerator.cpp
PVSlidersGroup.cpp
PVSlidersManager.cpp
PVZoneRendering.cpp
PVZoneTree.cpp
PVZoneTreeBase.cpp
PVZonesManager.cpp
PVZonesProcessor.cpp
PVZoomAxisSlider.cpp
PVZoomAxisSliders.cpp
PVZoomedParallelScene.cpp
PVZoomedParallelView.cpp
PVZoomedParallelViewSelectionLine.cpp
PVZoomedParallelViewParamsWidget.cpp
PVZoomedSelectionAxisSlider.cpp
PVZoomedSelectionAxisSliders.cpp
PVZoomedZoneTree.cpp

# Display plugins
PVDisplayViewFullParallel.cpp
PVDisplayViewZoomedParallel.cpp
PVDisplayViewHitCount.cpp
PVDisplayViewScatter.cpp
PVDisplayViewTimeseries.cpp

# ZoomableDrawingArea & co
PVZoomableDrawingArea.cpp
PVZoomableDrawingAreaWithAxes.cpp

PVSelectionRectangle.cpp
PVSelectionRectangleItem.cpp
PVSelectionHandleItem.cpp

PVSelectionRectangleInteractor.cpp

PVZoomableDrawingAreaInteractor.cpp
PVZoomableDrawingAreaInteractorMajorY.cpp

PVZoomableDrawingAreaConstraintsHomothetic.cpp
PVZoomableDrawingAreaConstraintsMajorY.cpp
PVZoomableDrawingAreaInteractorHomothetic.cpp

# views
PVHitCountView.cpp
PVHitCountViewBackend.cpp
PVHitCountViewInteractor.cpp
PVHitCountViewParamsWidget.cpp
PVHitCountViewSelectionRectangle.cpp

PVScatterView.cpp
PVScatterViewBackend.cpp
PVScatterViewInteractor.cpp
PVScatterViewImage.cpp
PVScatterViewImagesManager.cpp
PVScatterViewDataInterface.cpp
PVScatterViewDataImpl.cpp
PVScatterViewSelectionRectangle.cpp
PVScatterViewParamsWidget.cpp

PVSeriesView.cpp
PVSeriesViewWidget.cpp
PVSeriesViewParamsWidget.cpp
PVSeriesViewZoomer.cpp
)

set(OPENCL_SRC_FILES
	PVBCIDrawingBackendOpenCL.cpp
	PVBCIBackendImageOpenCL.cpp
)

set(MOC_HEADERS
../include/pvparallelview/PVRenderingJob.h
../include/pvparallelview/PVFullParallelView.h
../include/pvparallelview/PVFullParallelScene.h
../include/pvparallelview/PVFullParallelViewParamsWidget.h
../include/pvparallelview/PVSelectionRectangleItem.h
../include/pvparallelview/PVSelectionRectangle.h
../include/pvparallelview/PVZoomedParallelViewSelectionLine.h
../include/pvparallelview/PVZoomedParallelScene.h
../include/pvparallelview/PVZoomedParallelView.h
../include/pvparallelview/PVSlidersGroup.h
../include/pvparallelview/PVAxisGraphicsItem.h
../include/pvparallelview/PVAxisLabel.h
../include/pvparallelview/PVAxisHeader.h
../include/pvparallelview/PVAbstractAxisSlider.h
../include/pvparallelview/PVAbstractAxisSliders.h
../include/pvparallelview/PVSelectionAxisSliders.h
../include/pvparallelview/PVZoomAxisSliders.h
../include/pvparallelview/PVZoomedSelectionAxisSliders.h
../include/pvparallelview/PVZoomedParallelViewParamsWidget.h
../include/pvparallelview/PVZoomableDrawingArea.h
../include/pvparallelview/PVHitCountView.h
../include/pvparallelview/PVHitCountViewParamsWidget.h
../include/pvparallelview/PVScatterView.h
../include/pvparallelview/PVScatterViewParamsWidget.h
../include/pvparallelview/PVSeriesView.h
../include/pvparallelview/PVSeriesViewWidget.h
../include/pvparallelview/PVSeriesViewParamsWidget.h
../include/pvparallelview/PVSeriesViewZoomer.h
../include/pvparallelview/PVSeriesTreeWidget.h
)

################################################
# Generate extra QT files (moc, uic, rcc)
################################################

qt6_wrap_cpp(MOC_SRC ${MOC_HEADERS} OPTIONS)

################################################
# Declare the new library and linkage options
################################################

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/bci_z24.h
  COMMAND ${squey_SOURCE_DIR}/scripts/obfuscate-ocl-kernel.sh ${CMAKE_CURRENT_SOURCE_DIR}/opencl/bci_z24.cl ${CMAKE_CURRENT_BINARY_DIR}/bci_z24.h
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/opencl/bci_z24.cl
)

list(APPEND SRC_FILES ${OPENCL_SRC_FILES} ${CMAKE_CURRENT_BINARY_DIR}/bci_z24.h)

# The hit graph kernels are built twice from the same .ipp : the avx2 unit needs its
# own instruction set, and neither may be folded into a unity blob that would drag the
# wider flags onto unrelated sources. The shared PCH is built without avx2, so it
# cannot be reused by the avx2 unit either.
#
# Off x86 there is no second instruction set to build for, and -mavx2 is not even a
# valid flag there : only the baseline unit is compiled, and PVHitGraphDataOMP.cpp
# aliases the avx2 namespace onto it so the dispatch still links.
if(SQUEY_TARGET_X86_64)
    set_source_files_properties(PVHitGraphDataOMP_kernels_avx2.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2" SKIP_UNITY_BUILD_INCLUSION ON
                   SKIP_PRECOMPILE_HEADERS ON)
else()
    list(REMOVE_ITEM SRC_FILES PVHitGraphDataOMP_kernels_avx2.cpp)
endif()
set_source_files_properties(PVHitGraphDataOMP_kernels_baseline.cpp
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON SKIP_PRECOMPILE_HEADERS ON)

add_library(pvparallelview SHARED ${SRC_FILES} ${MOC_SRC})
set_target_properties(pvparallelview PROPERTIES SOVERSION 1)
squey_speedup_compilation(pvparallelview QT)

target_include_directories(pvparallelview
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
	PRIVATE
		${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(pvparallelview
	PUBLIC pvkernel pvsquey pvdisplays
	       Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Xml KF6::ItemModels
	PRIVATE ${TBB_LIBRARIES} ${Boost_LIBRARIES} ${OpenMP_LIBRARIES} Squey::OpenCL)

install(TARGETS pvparallelview LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
						RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR})
