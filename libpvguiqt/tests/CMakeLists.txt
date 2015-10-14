#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

###################################################################
# Add include directories for picviz's libs and external projects
###################################################################

include_directories(AFTER ${PVKERNEL_INCLUDE_DIR})
include_directories(AFTER ${PVHIVE_INCLUDE_DIR})
include_directories(AFTER ${PVGUIQT_INCLUDE_DIR})
include_directories(AFTER ${PVPARALLELVIEW_INCLUDE_DIRS})

add_subdirectory(squish)

set(COMMON_LIBRARIES pvparallelview pvguiqt pvhive picviz pvkernel ${TBB_LIBRARIES} ${BOOST_LIBRARIES} ${Boost_LIBRARIES_PYTHON} ${PYTHON_LIBRARY} ${OpenMP_LIBRARIES})

add_pv_compile_test(Tqt_datatree LINKS ${COMMON_LIBRARIES} FILES datatree-model.cpp)
add_pv_compile_test(Tqt_picviz_tree LINKS ${COMMON_LIBRARIES} FILES picviz_tree.cpp common.cpp)

qt5_wrap_cpp(AXES_COMB_MOC_SRC axes-comb_dlg.h OPTIONS ${QT5_WRAP_OPTIONS})
add_pv_compile_test(Tqt_axes_comb LINKS ${COMMON_LIBRARIES} FILES axes-comb.cpp ${AXES_COMB_MOC_SRC})
add_pv_compile_test(Tqt_axes_comb_view LINKS ${COMMON_LIBRARIES} FILES axes-comb-view.cpp ${AXES_COMB_MOC_SRC})
add_pv_compile_test(Tqt_listing_model LINKS ${COMMON_LIBRARIES} FILES listing_model.cpp common.cpp)

qt5_wrap_cpp(STATS_LISTING_MOC_SRC stats_listing.h OPTIONS ${QT5_WRAP_OPTIONS})
add_pv_compile_test(Tqt_stats_listing LINKS ${COMMON_LIBRARIES} FILES stats_listing.cpp common.cpp ${STATS_LISTING_MOC_SRC})
add_pv_compile_test(Tqt_layerstack LINKS ${COMMON_LIBRARIES} FILES layerstack-model.cpp common.cpp)
add_pv_compile_test(Tqt_layerstack_widget LINKS ${COMMON_LIBRARIES} FILES layerstack-widget.cpp common.cpp)
add_pv_compile_test(Tqt_huge-model LINKS ${COMMON_LIBRARIES} FILES huge-model.cpp)
add_pv_compile_test(Tqt_css LINKS ${COMMON_LIBRARIES} FILES css.cpp)
add_pv_compile_test(Tqt_desktop_geometry LINKS ${COMMON_LIBRARIES} FILES desktop_geometry.cpp)
add_pv_compile_test(Tqt_vertical_tabs LINKS ${COMMON_LIBRARIES} FILES vertical_tabs.cpp common.cpp)
add_pv_compile_test(Tqt_import_source_to_project_dlg LINKS ${COMMON_LIBRARIES} FILES import_source_to_project_dlg.cpp)
add_pv_compile_test(Tqt_displays_plugins LINKS ${COMMON_LIBRARIES} FILES displays_plugins.cpp common.cpp)

# Workspaces
if (CUDA_FOUND)
	qt5_wrap_cpp(WORKSPACES_MOC_SRC workspaces.h OPTIONS ${QT5_WRAP_OPTIONS})
	set(LINK_LIBRARIES ${LINK_LIBRARIES} ${CUDA_LIBRARIES})
	include_directories(AFTER ${CUDA_INCLUDE_DIRS})

	add_pv_compile_test(Tqt_workspaces LINKS ${MOC_SRC} ${MOC_HEADERS} ${COMMON_LIBRARIES} ${CUDA_LIBRARIES} ${LINK_LIBRARIES} FILES workspaces.cpp common.cpp)
endif(CUDA_FOUND)
