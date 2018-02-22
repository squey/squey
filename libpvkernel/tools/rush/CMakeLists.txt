#
# @file
#
# @copyright (C) ESI Group INENDI April 2016

set(LINK_LIBRARIES ${PVLICENSE_LIBRARIES} pvkernel)
include_directories(${PVLICENSE_INCLUDE_DIR})
add_definitions(-fopenmp)

#inendi-logstats
message(STATUS "PVCOP_LIBRARIES=${PVCOP_LIBRARIES}")
compile_and_install_tool(inendi-logstats LINKS ${LINK_LIBRARIES} ${PVCOP_LIBRARIES} ${PVLOGGER_LIBRARIES} FILES log_stats.cpp)