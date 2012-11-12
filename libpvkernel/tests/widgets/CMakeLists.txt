#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

# Test data-tree model
add_executable(Twidgets_datatreemodel datatree-model.cpp)
target_link_libraries(Twidgets_datatreemodel pvkernel ${QT_LIBRARIES})

# Test data-tree proxy
add_executable(Twidgets_datatreeproxy datatree-proxy.cpp)
target_link_libraries(Twidgets_datatreeproxy pvkernel ${QT_LIBRARIES})

# Color picker
add_executable(Twidgets_color-picker color-picker.cpp)
target_link_libraries(Twidgets_color-picker pvkernel ${QT_LIBRARIES})

# Color dialog
add_executable(Twidgets_color-dialog color-dialog.cpp)
target_link_libraries(Twidgets_color-dialog pvkernel ${QT_LIBRARIES})
