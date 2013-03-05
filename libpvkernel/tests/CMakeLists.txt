#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

add_subdirectory(core)
add_subdirectory(filter)
add_subdirectory(rush)
add_subdirectory(widgets)

propagate_tests_targets_to_parent()
