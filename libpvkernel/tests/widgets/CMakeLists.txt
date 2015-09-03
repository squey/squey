#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

# Test data-tree model
declare_pv_test(Twidgets_datatreemodel datatree-model.cpp)
target_link_libraries(Twidgets_datatreemodel pvkernel)

# Test data-tree proxy
declare_pv_test(Twidgets_datatreeproxy datatree-proxy.cpp)
target_link_libraries(Twidgets_datatreeproxy pvkernel)

# Color picker
declare_pv_test(Twidgets_color-picker color-picker.cpp)
target_link_libraries(Twidgets_color-picker pvkernel)

# Color dialog
declare_pv_test(Twidgets_color-dialog color-dialog.cpp)
target_link_libraries(Twidgets_color-dialog pvkernel)

# QGraphicsView reimplementation to use QScrollBar64
QT5_WRAP_CPP(PVGV_MOC mygraphicsview.h OPTIONS ${QT5_WRAP_OPTIONS})
declare_pv_test(Epvgraphicsview pvgraphicsview.cpp ${PVGV_MOC})
target_link_libraries(Epvgraphicsview pvkernel)

declare_pv_test(Epvgraphicsview_inv_y graphicsview_inv_y.cpp)
target_link_libraries(Epvgraphicsview_inv_y pvkernel)

declare_lib_tests_target()
