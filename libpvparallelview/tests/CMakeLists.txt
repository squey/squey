#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

include_directories(AFTER ${PVPARALLELVIEW_INCLUDE_DIRS})
add_definitions(${INENDI_DEFINITIONS})
include_directories(AFTER ${INENDI_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Core_EXECUTABLE_COMPILE_FLAGS}")

set(CUDA_PROPAGATE_HOST_FLAGS "OFF")
set(CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30" "-Xptxas=-v" "-Xcompiler=-fPIC")
#set(CUDA_NVCC_FLAGS "-g" "-G")

set(LINK_LIBRARIES ${INENDI_LIBRARY} ${PVKERNEL_LIBRARY} ${TBB_LIBRARIES} ${ICU_LIBRARY} pvparallelview ${OpenMP_LIBRARIES} numa)

QT5_WRAP_CPP(ZOOM_DLG_MOC zoom_dlg.h)

add_pv_compile_test(full_parallel_view LINKS ${LINK_LIBRARIES} FILES full_parallel_view.cpp common.cpp)
add_pv_compile_test(both_views LINKS ${LINK_LIBRARIES} FILES both_views.cpp ${ZOOM_DLG_MOC} common.cpp)
add_pv_compile_test(zoomed_parallel_view LINKS ${LINK_LIBRARIES} FILES zoomed_parallel_view.cpp common.cpp)
add_pv_compile_test(create_tree LINKS ${LINK_LIBRARIES} FILES create_tree.cpp)
add_pv_compile_test(Pqt-extract-sse LINKS ${LINK_LIBRARIES} FILES qt-extract-sse.cpp)
add_pv_compile_test(Eqt-tbb-tasks LINKS ${LINK_LIBRARIES} FILES qt-tbb-tasks.cpp)
add_pv_compile_test(Eqt-tbb-traversal LINKS ${LINK_LIBRARIES} FILES qt-tbb-traversal.cpp)
add_pv_compile_test(sse_bci LINKS ${LINK_LIBRARIES} FILES sse_bci.cpp)
add_pv_compile_test(create_quadtree LINKS ${LINK_LIBRARIES} FILES create_quadtree.cpp)
add_pv_compile_test(create_zoomedzonetree LINKS ${LINK_LIBRARIES} FILES create_zoomedzonetree.cpp)
add_pv_compile_test(extract_quadtree LINKS ${LINK_LIBRARIES} FILES extract_quadtree.cpp)
add_pv_compile_test(Tquadtree_impl LINKS ${LINK_LIBRARIES} FILES quadtree_impl.cpp)
add_pv_compile_test(Tmultigrid_impl LINKS ${LINK_LIBRARIES} FILES multigrid_impl.cpp)
add_pv_compile_test(show_codes_gl LINKS ${LINK_LIBRARIES} FILES show_codes_gl.cpp)
add_pv_compile_test(masks LINKS ${LINK_LIBRARIES} FILES masks.cpp)
add_pv_compile_test(Ppview_zoomed_zone_tree_perf LINKS ${LINK_LIBRARIES} FILES zoomed_zone_tree_perf.cpp common.cpp)
add_pv_compile_test(Tpview_dump_zoomed_zone_tree LINKS ${LINK_LIBRARIES} FILES dump_zoomed_zone_tree.cpp common.cpp)
add_pv_compile_test(Tpview_dump_zone_tree LINKS ${LINK_LIBRARIES} FILES dump_zone_tree.cpp common.cpp)
add_pv_compile_test(Tpview_dump_all_trees LINKS ${LINK_LIBRARIES} FILES dump_all_trees.cpp common.cpp)
add_pv_compile_test(Ehitgraph-test LINKS ${LINK_LIBRARIES} FILES hitgraph-test.cpp common.cpp)
add_pv_compile_test(Ehitgraph-zonetree LINKS ${LINK_LIBRARIES} FILES hitgraph-zonetree.cpp common.cpp)
add_pv_compile_test(Ehitgraph-perf LINKS ${LINK_LIBRARIES} FILES hitgraph-perf.cpp)
add_pv_compile_test(Ehitgraph-alpha-perf LINKS ${LINK_LIBRARIES} FILES hitgraph-alpha-perf.cpp)
add_pv_compile_test(Ehist_plotted_perf LINKS ${LINK_LIBRARIES} FILES hist_plotted_perf.cpp common.cpp)
add_pv_compile_test(zoomable_drawing_area_test LINKS ${LINK_LIBRARIES} FILES zoomable_drawing_area_test.cpp)
add_pv_compile_test(hitgraph-buffer LINKS ${LINK_LIBRARIES} FILES hitgraph-buffer.cpp)
add_pv_compile_test(hit_count_view LINKS ${LINK_LIBRARIES} FILES hit_count_view.cpp common.cpp)
add_pv_compile_test(hit_count_sel_perf LINKS ${LINK_LIBRARIES} FILES hit_count_sel_perf.cpp common.cpp)
add_pv_compile_test(scatter_view LINKS ${LINK_LIBRARIES} FILES scatter_view.cpp common.cpp)
add_pv_compile_test(Tqt_random_points LINKS ${LINK_LIBRARIES} FILES random_points.cpp)

add_pv_compile_test(Tpview_zone_tree_dump_load LINKS ${LINK_LIBRARIES} FILES zone_tree_dump_load.cpp common.cpp)
add_test(Tpview_zone_tree_dump_load Tpview_zone_tree_dump_load 0 50 2)

add_pv_compile_test(Tpview_quadtree_dump_load LINKS ${LINK_LIBRARIES} FILES quadtree_dump_load.cpp)
add_test(Tpview_quadtree_dump_load Tpview_quadtree_dump_load 1000000 42)

add_pv_compile_test(Tpview_zoomed_zone_tree_dump_load LINKS ${LINK_LIBRARIES} FILES zoomed_zone_tree_dump_load.cpp common.cpp)
add_test(Tpview_zoomed_zone_tree_dump_load Tpview_zoomed_zone_tree_dump_load 0 100 2)

if (CUDA_FOUND)
	include_directories(AFTER ${CUDA_INCLUDE_DIRS})

	add_pv_compile_test(show_codes_cuda_lib LINKS ${LINK_LIBRARIES} FILES show_codes_cuda_lib.cpp helpers.cpp)
	add_pv_compile_test(perf_codes_cuda_lib LINKS ${LINK_LIBRARIES} FILES perf_codes_cuda_lib.cpp helpers.cpp)
endif (CUDA_FOUND)
