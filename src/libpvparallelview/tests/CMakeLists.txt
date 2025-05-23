#
# @file
#

include_directories(AFTER ${PVPARALLELVIEW_INCLUDE_DIRS})
add_definitions(${SQUEY_DEFINITIONS})
include_directories(AFTER ${SQUEY_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt6Core_EXECUTABLE_COMPILE_FLAGS}")

set(LINK_LIBRARIES ${SQUEY_LIBRARY} ${PVKERNEL_LIBRARY} pvparallelview)
include_directories(${PVLICENSE_INCLUDE_DIR})

add_squey_compile_test(zoomed_parallel_view LINKS ${LINK_LIBRARIES} FILES zoomed_parallel_view.cpp)
add_squey_compile_test(zoomable_drawing_area_test LINKS ${LINK_LIBRARIES} FILES zoomable_drawing_area_test.cpp)

add_squey_compile_test(Tquadtree_impl LINKS ${LINK_LIBRARIES} FILES quadtree_impl.cpp)
add_squey_test(Tquadtree_impl Tquadtree_impl 10 10)

add_squey_compile_test(Thitgraph-buffer LINKS ${LINK_LIBRARIES} FILES hitgraph-buffer.cpp)
add_squey_test(Thitgraph-buffer Thitgraph-buffer)

add_squey_compile_test(Tpview_zone_tree_dump_load LINKS ${LINK_LIBRARIES} FILES zone_tree_dump_load.cpp)
add_squey_test(Tpview_zone_tree_dump_load Tpview_zone_tree_dump_load 0 50 2)

add_squey_compile_test(Tpview_quadtree_dump_load LINKS ${LINK_LIBRARIES} FILES quadtree_dump_load.cpp)
add_squey_test(Tpview_quadtree_dump_load Tpview_quadtree_dump_load 1000000 42)

add_squey_compile_test(Tpview_zoomed_zone_tree_dump_load LINKS ${LINK_LIBRARIES} FILES zoomed_zone_tree_dump_load.cpp)
add_squey_test(Tpview_zoomed_zone_tree_dump_load Tpview_zoomed_zone_tree_dump_load 0 100 2)

add_squey_compile_test(Tpview_square_selection LINKS ${LINK_LIBRARIES} FILES square_selection.cpp)
add_squey_test(Tpview_square_selection Tpview_square_selection)
add_bench(Tpview_square_selection LINKS ${LINK_LIBRARIES} FILES square_selection.cpp)

add_squey_compile_test(Tpview_zone_tree_create LINKS ${LINK_LIBRARIES} FILES zone_tree_create.cpp)
add_squey_test(Tpview_zone_tree_create Tpview_zone_tree_create)
add_bench(Tpview_zone_tree_create LINKS ${LINK_LIBRARIES} FILES zone_tree_create.cpp)

add_squey_compile_test(Tpview_zone_tree_filter LINKS ${LINK_LIBRARIES} FILES zone_tree_filter.cpp)
add_squey_test(Tpview_zone_tree_filter Tpview_zone_tree_filter)
add_bench(Tpview_zone_tree_filter LINKS ${LINK_LIBRARIES} FILES zone_tree_filter.cpp)

add_squey_compile_test(Topencl_build_test LINKS ${LINK_LIBRARIES} FILES opencl_build_test.cpp)
add_squey_test(Topencl_build_test Topencl_build_test)

add_squey_compile_test(bcicodes_opencl LINKS ${LINK_LIBRARIES} FILES bcicodes_opencl.cpp bci_helpers.cpp)
add_bench(bcicodes_opencl LINKS ${LINK_LIBRARIES} FILES bcicodes_opencl.cpp bci_helpers.cpp)

add_squey_compile_test(Trendering_pipeline_base LINKS ${LINK_LIBRARIES} FILES rendering_pipeline_base.cpp)
add_squey_test(Trendering_pipeline_base Trendering_pipeline_base 1000)
add_bench(rendering_pipeline_base LINKS ${LINK_LIBRARIES} FILES rendering_pipeline_base.cpp)

#GL Timeseries
# add_squey_compile_test(Tgl_timeseries LINKS ${LINK_LIBRARIES} Qt6OpenGL FILES gl_timeseries.cpp)
# #add_squey_test(Tgl_timeseries Tgl_timeseries)
# add_bench(gl_timeseries LINKS ${LINK_LIBRARIES} FILES gl_timeseries.cpp)

add_squey_compile_test(Tseries_zoomer LINKS ${LINK_LIBRARIES} FILES series_zoomer.cpp)
#add_squey_test(Tseries_zoomer Tseries_zoomer)
add_bench(series_zoomer LINKS ${LINK_LIBRARIES} FILES series_zoomer.cpp)

file(RELATIVE_PATH REL_TEST_DIR ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake DESTINATION "tests/${REL_TEST_DIR}" COMPONENT tests)