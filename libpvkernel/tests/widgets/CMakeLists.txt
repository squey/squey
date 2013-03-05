#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

# Test data-tree model
declare_pv_test(Twidgets_datatreemodel datatree-model.cpp)
target_link_libraries(Twidgets_datatreemodel pvkernel ${QT_LIBRARIES})

# Test data-tree proxy
declare_pv_test(Twidgets_datatreeproxy datatree-proxy.cpp)
target_link_libraries(Twidgets_datatreeproxy pvkernel ${QT_LIBRARIES})

# Color picker
declare_pv_test(Twidgets_color-picker color-picker.cpp)
target_link_libraries(Twidgets_color-picker pvkernel ${QT_LIBRARIES})

# Color dialog
declare_pv_test(Twidgets_color-dialog color-dialog.cpp)
target_link_libraries(Twidgets_color-dialog pvkernel ${QT_LIBRARIES})

# QGraphicsView reimplementation to use QScrollBar64
QT4_WRAP_CPP(PVGV_MOC mygraphicsview.h OPTIONS ${QT4_WRAP_OPTIONS})
declare_pv_test(Epvgraphicsview pvgraphicsview.cpp ${PVGV_MOC})
target_link_libraries(Epvgraphicsview pvkernel ${QT_LIBRARIES})

