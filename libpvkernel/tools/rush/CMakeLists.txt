#
# @file
#

set(LINK_LIBRARIES ${PVLICENSE_LIBRARIES} pvkernel)
include_directories(${PVLICENSE_INCLUDE_DIR})
add_definitions(-fopenmp)

#squey-logstats
#compile_and_install_tool(squey-logstats LINKS ${LINK_LIBRARIES} ${PVCOP_LIBRARIES} ${PVLOGGER_LIBRARIES} FILES log_stats.cpp)