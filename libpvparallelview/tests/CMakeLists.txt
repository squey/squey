#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libparallelview/src/include)
add_definitions(${PICVIZ_DEFINITIONS})
include_directories(AFTER ${PICVIZ_INCLUDE_DIRS})

set(CUDA_PROPAGATE_HOST_FLAGS "OFF")
set(CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30" "-Xptxas=-v")
#set(CUDA_NVCC_FLAGS "-g" "-G")

add_definitions(${OpenMP_CXX_FLAGS})
set(QT_USE_QTOPENGL TRUE)
include(${QT_USE_FILE})

set(LINK_LIBRARIES ${PICVIZ_LIBRARY} ${PVKERNEL_LIBRARY} ${QT_LIBRARIES} ${TBB_LIBRARIES} ${ICU_LIBRARY} pvparallelview ${OpenMP_LIBRARIES})

enable_testing()

add_executable(full_parallel_view full_parallel_view.cpp common.cpp)
target_link_libraries(full_parallel_view ${LINK_LIBRARIES})

add_executable(full_parallel_view_selections_sync full_parallel_view_selections_sync.cpp common.cpp)
target_link_libraries(full_parallel_view_selections_sync ${LINK_LIBRARIES})

QT4_WRAP_CPP(ZOOM_DLG_MOC zoom_dlg.h OPTIONS ${QT4_WRAP_OPTIONS})
add_executable(both_views both_views.cpp ${ZOOM_DLG_MOC} common.cpp)
target_link_libraries(both_views ${LINK_LIBRARIES})

add_executable(zoomed_parallel_view zoomed_parallel_view.cpp)
target_link_libraries(zoomed_parallel_view ${LINK_LIBRARIES})

add_executable(zoomed_zone_tree_render zoomed_zone_tree_render.cpp)
target_link_libraries(zoomed_zone_tree_render ${LINK_LIBRARIES})

add_executable(zoomed_zone_tree_render2 zoomed_zone_tree_render2.cpp)
target_link_libraries(zoomed_zone_tree_render2 ${LINK_LIBRARIES})

add_executable(create_tree create_tree.cpp)
target_link_libraries(create_tree ${LINK_LIBRARIES})

add_executable(quadtree-create-bench quadtree-create-bench.cpp common.cpp)
target_link_libraries(quadtree-create-bench ${LINK_LIBRARIES})

add_executable(sse_bci sse_bci.cpp)
target_link_libraries(sse_bci ${LINK_LIBRARIES})

add_executable(create_quadtree create_quadtree.cpp)
target_link_libraries(create_quadtree ${LINK_LIBRARIES})

add_executable(create_zoomedzonetree create_zoomedzonetree.cpp)
target_link_libraries(create_zoomedzonetree ${LINK_LIBRARIES})

add_executable(extract_quadtree extract_quadtree.cpp)
target_link_libraries(extract_quadtree ${LINK_LIBRARIES})

add_executable(quadtree_browse quadtree_browse.cpp)
target_link_libraries(quadtree_browse ${LINK_LIBRARIES})

add_executable(Tquadtree_impl quadtree_impl.cpp)
target_link_libraries(Tquadtree_impl ${LINK_LIBRARIES})

add_executable(Tmultigrid_impl multigrid_impl.cpp)
target_link_libraries(Tmultigrid_impl ${LINK_LIBRARIES})

add_executable(show_codes_gl show_codes_gl.cpp)
target_link_libraries(show_codes_gl ${LINK_LIBRARIES})

add_executable(masks masks.cpp)
target_link_libraries(masks ${LINK_LIBRARIES})

#add_executable(Tsort_test sort_test.cpp)
#target_link_libraries(Tsort_test ${LINK_LIBRARIES})

#add_executable(qgraphicsitem_events qgraphicsitem_events.cpp)
#target_link_libraries(qgraphicsitem_events ${LINK_LIBRARIES})

if (CUDA_FOUND)
	include_directories(AFTER ${CUDA_INCLUDE_DIRS})
	cuda_compile(CUDA_KERNELS bci_cuda.cu)
	add_executable(show_codes_cuda show_codes_cuda.cpp ${CUDA_KERNELS})
	target_link_libraries(show_codes_cuda ${LINK_LIBRARIES} ${CUDA_LIBRARIES})

# does not compile
#	cuda_compile(CUDA_KERNELS bci_cuda_z24.cu)
#	add_executable(show_codes_cuda24 show_codes_cuda.cpp ${CUDA_KERNELS})
#	target_link_libraries(show_codes_cuda24 ${LINK_LIBRARIES} ${CUDA_LIBRARIES})

# does not compile
#	add_executable(show_codes_cuda_tex show_codes_cuda_tex.cpp ${CUDA_KERNELS})
#	target_link_libraries(show_codes_cuda_tex ${LINK_LIBRARIES} ${CUDA_LIBRARIES})

	cuda_compile(CUDA_KERNELS bci_cuda_z24_atomic.cu)
	add_executable(show_codes_cuda24_atomic show_codes_cuda.cpp ${CUDA_KERNELS})
	target_link_libraries(show_codes_cuda24_atomic ${LINK_LIBRARIES} ${CUDA_LIBRARIES})

	#cuda_compile(CUDA_KERNELS bci_cuda_z24_unroll_x.cu)
	#add_executable(show_codes_cuda24_unroll_x show_codes_cuda.cpp ${CUDA_KERNELS})
	#target_link_libraries(show_codes_cuda24_unroll_x ${LINK_LIBRARIES} ${CUDA_LIBRARIES})

	cuda_compile(CUDA_KERNELS bci_cuda_z24_s16.cu)
	add_executable(show_codes_cuda24_s16 show_codes_cuda.cpp ${CUDA_KERNELS})
	target_link_libraries(show_codes_cuda24_s16 ${LINK_LIBRARIES} ${CUDA_LIBRARIES})

	cuda_compile(CUDA_KERNELS bci_cuda_z24_smembci.cu)
	add_executable(show_codes_cuda24_smembci show_codes_cuda.cpp ${CUDA_KERNELS})
	target_link_libraries(show_codes_cuda24_smembci ${LINK_LIBRARIES} ${CUDA_LIBRARIES})

	cuda_compile(CUDA_KERNELS multiple_zones_cuda.cu)
	add_executable(multiple_zones_cuda multiple_zones_cuda.cpp ${CUDA_KERNELS})
	target_link_libraries(multiple_zones_cuda ${LINK_LIBRARIES} ${CUDA_LIBRARIES})

	add_executable(lines_drawing lines_drawing.cpp)
	target_link_libraries(lines_drawing ${LINK_LIBRARIES})

	add_executable(show_codes_cuda_lib show_codes_cuda_lib.cpp)
	target_link_libraries(show_codes_cuda_lib ${LINK_LIBRARIES})
endif (CUDA_FOUND)
