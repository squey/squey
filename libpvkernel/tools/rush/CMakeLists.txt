#
# @file
#
# @copyright (C) ESI Group INENDI April 2016

set(LINK_LIBRARIES pvkernel ${TBB_LIBRARIES} ${ICU_LIBRARY})
add_definitions(-fopenmp)

#inendi-logstats
compile_and_install_tool(inendi-logstats LINKS ${LINK_LIBRARIES} FILES log_stats.cpp)