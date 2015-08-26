#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

add_subdirectory(layer-filters)
add_subdirectory(mapping-filters)
add_subdirectory(plotting-filters)
if(ENABLE_CORRELATION)
	add_subdirectory(row-filters)
endif()
add_subdirectory(axis-computation)
add_subdirectory(sorting-functions)

