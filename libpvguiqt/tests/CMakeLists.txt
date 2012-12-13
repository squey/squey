#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

include_directories(AFTER ${PVKERNEL_INCLUDE_DIR})
include_directories(AFTER ${PVHIVE_INCLUDE_DIR})
include_directories(AFTER ${PVGUIQT_INCLUDE_DIR})
include_directories(AFTER ${CMAKE_SOURCE_DIR}/libpvparallelview/src/include)

add_subdirectory(squish)

set(COMMON_LIBRARIES pvparallelview pvguiqt pvhive picviz pvkernel ${TBB_LIBRARIES} ${BOOST_LIBRARIES} ${Boost_LIBRARIES_PYTHON} ${PYTHON_LIBRARY} ${OpenMP_LIBRARIES})

# Datatree model
add_executable(Tqt_datatree datatree-model.cpp)
target_link_libraries(Tqt_datatree ${COMMON_LIBRARIES})

# Picviz datatree model
add_executable(Tqt_picviz_tree picviz_tree.cpp common.cpp)
target_link_libraries(Tqt_picviz_tree ${COMMON_LIBRARIES})

# Axes combination model
qt4_wrap_cpp(AXES_COMB_MOC_SRC axes-comb_dlg.h OPTIONS ${QT4_WRAP_OPTIONS})
add_executable(Tqt_axes_comb axes-comb.cpp ${AXES_COMB_MOC_SRC})
target_link_libraries(Tqt_axes_comb ${COMMON_LIBRARIES})

add_executable(Tqt_axes_comb_view axes-comb-view.cpp ${AXES_COMB_MOC_SRC})
target_link_libraries(Tqt_axes_comb_view ${COMMON_LIBRARIES})

# Picviz listing model
add_executable(Tqt_listing_model listing_model.cpp common.cpp)
target_link_libraries(Tqt_listing_model ${COMMON_LIBRARIES})

# Stats listing
add_executable(Tqt_stats_listing stats_listing.cpp common.cpp)
target_link_libraries(Tqt_stats_listing ${COMMON_LIBRARIES})

# Picviz layer-stack model
add_executable(Tqt_layerstack layerstack-model.cpp common.cpp)
target_link_libraries(Tqt_layerstack ${COMMON_LIBRARIES})

# Picviz layer-stack widget 
add_executable(Tqt_layerstack_widget layerstack-widget.cpp common.cpp)
target_link_libraries(Tqt_layerstack_widget ${COMMON_LIBRARIES})

# Huge model
add_executable(Tqt_huge-model huge-model.cpp)
target_link_libraries(Tqt_huge-model ${COMMON_LIBRARIES})

# Confusing Qt Stylesheets
add_executable(Tqt_css css.cpp)
target_link_libraries(Tqt_css ${COMMON_LIBRARIES})

# Workspaces
if (CUDA_FOUND)
	qt4_wrap_cpp(WORKSPACES_MOC_SRC workspaces.h OPTIONS ${QT4_WRAP_OPTIONS})
	set(LINK_LIBRARIES ${LINK_LIBRARIES} ${CUDA_LIBRARIES} /opt/cuda/lib64/libnpp.so)
	include_directories(AFTER ${CUDA_INCLUDE_DIRS})
	add_executable(Tqt_workspaces workspaces.cpp common.cpp)
	#add_executable(Tqt_workspaces workspaces.cpp common.cpp ${WORKSPACES_MOC_SRC})
	target_link_libraries(Tqt_workspaces ${MOC_SRC} ${MOC_HEADERS} ${COMMON_LIBRARIES} ${CUDA_LIBRARIES} ${LINK_LIBRARIES} )
endif(CUDA_FOUND)

# Desktop geometry
add_executable(Tqt_desktop_geometry desktop_geometry.cpp)
target_link_libraries(Tqt_desktop_geometry ${COMMON_LIBRARIES})

# Vertical tabs
add_executable(Tqt_vertical_tabs vertical_tabs.cpp common.cpp)
target_link_libraries(Tqt_vertical_tabs ${COMMON_LIBRARIES})

# PVImportSourceToProjectDlg
add_executable(Tqt_import_source_to_project_dlg import_source_to_project_dlg.cpp)
target_link_libraries(Tqt_import_source_to_project_dlg ${COMMON_LIBRARIES})

# correlation menu
add_executable(Tqt_correlation_menu correlation_menu.cpp)
target_link_libraries(Tqt_correlation_menu ${COMMON_LIBRARIES})

# Displays plugins
add_executable(Tqt_displays_plugins displays_plugins.cpp common.cpp)
target_link_libraries(Tqt_displays_plugins ${COMMON_LIBRARIES})


# AD2G Widget
add_executable(Tqt_AD2GWidget AD2GWidget.cpp)
target_link_libraries(Tqt_AD2GWidget ${COMMON_LIBRARIES})

# Listing filter performance
add_executable(Pqt_listing_sel_filter listing_sel_filter_perfs.cpp)
target_link_libraries(Pqt_listing_sel_filter ${COMMON_LIBRARIES})
