#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

###################################################################
# Add include directories for inendi's libs and external projects
###################################################################

include_directories(AFTER ${PVKERNEL_INCLUDE_DIR})
include_directories(AFTER ${PVGUIQT_INCLUDE_DIR})
include_directories(AFTER ${PVPARALLELVIEW_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Core_EXECUTABLE_COMPILE_FLAGS}")

set(COMMON_LIBRARIES pvparallelview pvguiqt ${PVLICENSE_LIBRARIES} inendi pvkernel)

add_inspector_compile_test(Tqt_listing_model LINKS ${COMMON_LIBRARIES} FILES listing_model.cpp common.cpp)

add_inspector_compile_test(Tqt_layerstack LINKS ${COMMON_LIBRARIES} FILES layerstack-model.cpp common.cpp)
add_inspector_compile_test(Tqt_layerstack_widget LINKS ${COMMON_LIBRARIES} FILES layerstack-widget.cpp common.cpp)
add_inspector_compile_test(Tqt_vertical_tabs LINKS ${COMMON_LIBRARIES} FILES vertical_tabs.cpp common.cpp)
add_inspector_compile_test(Tqt_displays_plugins LINKS ${COMMON_LIBRARIES} FILES displays_plugins.cpp common.cpp)

add_inspector_compile_test(Tpvguiqt_sort_filter LINKS ${COMMON_LIBRARIES} FILES sort_filter.cpp)
add_inspector_test(Tpvguiqt_sort_filter Tpvguiqt_sort_filter)
add_bench(Tpvguiqt_sort_filter LINKS ${COMMON_LIBRARIES} FILES sort_filter.cpp)
