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
PVAntialiasingAction.cpp
PVAxisGraphicsItem.cpp
PVAxisLabel.cpp
PVAxisHeader.cpp
PVBCIBuffers.cpp
PVBCIBackendImageQPainter.cpp
PVBCIDrawingBackendQPainter.cpp
PVFloatingToolBar.cpp
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
PVDisplayViewScatterThumbnails.cpp
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

PVScatterThumbnail.cpp
PVScatterThumbnailsModel.cpp
PVScatterThumbnailsView.cpp

PVSeriesView.cpp
PVSeriesViewWidget.cpp
PVSeriesViewParamsWidget.cpp
PVSeriesViewZoomer.cpp
PVSeriesRendererRaster.cpp
PVSeriesRendererHybrid.cpp
PVSeriesSampleRows.cpp
)

set(OPENCL_SRC_FILES
	PVBCIDrawingBackendOpenCL.cpp
	PVBCIBackendImageOpenCL.cpp
)

# QRhi lives in QtGui, but behind the private headers: it is a supported API with no
# source compatibility promise, so the GPU backend is only built where they are actually
# installed, and PVSeriesView falls through to the CPU rasteriser where they are not.
find_package(Qt6 QUIET COMPONENTS GuiPrivate)
set(SQUEY_SERIES_QRHI OFF)
if(TARGET Qt6::GuiPrivate)
	# The private include directories only exist as generator expressions, so the header
	# is looked for where Qt puts it rather than asked for.
	find_path(SQUEY_QRHI_INCLUDE_DIR NAMES rhi/qrhi.h
	          PATHS ${Qt6Gui_INCLUDE_DIRS}
	          PATH_SUFFIXES ${Qt6Gui_VERSION}/QtGui
	          NO_DEFAULT_PATH)
	if(SQUEY_QRHI_INCLUDE_DIR)
		set(SQUEY_SERIES_QRHI ON)
	endif()
endif()
if(SQUEY_SERIES_QRHI AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/series.vert.qsb")
	message(STATUS "Series QRhi backend: baked shaders are missing, disabling")
	set(SQUEY_SERIES_QRHI OFF)
endif()
message(STATUS "Series QRhi backend: ${SQUEY_SERIES_QRHI}")

# The baked shaders are committed rather than built: qsb comes from qtshadertools, which
# this Qt is configured without, and a .qsb carries every dialect at once so the one file
# serves all platforms. Regenerate with `ninja series_shaders` after editing the GLSL,
# wherever a qsb happens to be at hand.
find_program(QSB_EXECUTABLE NAMES qsb qsb6)
if(QSB_EXECUTABLE)
	set(_shader_dialects --glsl "300es,330" --hlsl 50 --msl 12)
	add_custom_target(series_shaders
		COMMAND ${QSB_EXECUTABLE} ${_shader_dialects}
		        -o "${CMAKE_CURRENT_SOURCE_DIR}/shaders/series.vert.qsb"
		           "${CMAKE_CURRENT_SOURCE_DIR}/shaders/series.vert"
		COMMAND ${QSB_EXECUTABLE} ${_shader_dialects}
		        -o "${CMAKE_CURRENT_SOURCE_DIR}/shaders/series.frag.qsb"
		           "${CMAKE_CURRENT_SOURCE_DIR}/shaders/series.frag"
		COMMENT "Baking the series view shaders"
		VERBATIM)
endif()

set(MOC_HEADERS
../include/pvparallelview/PVRenderingJob.h
../include/pvparallelview/PVFullParallelView.h
../include/pvparallelview/PVFullParallelScene.h
../include/pvparallelview/PVFloatingToolBar.h
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
../include/pvparallelview/PVScatterThumbnailsModel.h
../include/pvparallelview/PVScatterThumbnailsView.h
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

if(SQUEY_SERIES_QRHI)
	QT6_ADD_RESOURCES(QRC_FILES_SRCS shaders/series_shaders.qrc)
	list(APPEND SRC_FILES PVSeriesRendererQRhi.cpp ${QRC_FILES_SRCS})
	set_source_files_properties(PVSeriesRendererQRhi.cpp PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

################################################
# Declare the new library and linkage options
################################################

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/bci_z24.h
  COMMAND ${CMAKE_COMMAND} -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/opencl/bci_z24.cl
          -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/bci_z24.h -DNAME=bci_z24_str
          -P ${squey_SOURCE_DIR}/cmake/EmbedSource.cmake
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/opencl/bci_z24.cl ${squey_SOURCE_DIR}/cmake/EmbedSource.cmake
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
target_compile_definitions(pvparallelview PRIVATE PVPARALLELVIEW_LIBRARY)
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

if(SQUEY_SERIES_QRHI)
	# PUBLIC so that whoever links the library sees the same set of backends it was built
	# with -- the tests pick the GPU one up from here. QRhi types reach the renderers'
	# public headers, so its include directories have to come with it.
	target_compile_definitions(pvparallelview PUBLIC SQUEY_SERIES_QRHI)
	target_link_libraries(pvparallelview PUBLIC Qt6::GuiPrivate)
endif()

install(TARGETS pvparallelview LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
						RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR})
