#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})

add_executable(log2csv log2csv.cpp)
target_link_libraries(log2csv ${PVKERNEL_LIBRARIES})
