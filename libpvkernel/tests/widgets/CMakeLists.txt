#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

# Test data-tree model
add_executable(Twidgets_datatreemodel datatree-model.cpp)
target_link_libraries(Twidgets_datatreemodel ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})
