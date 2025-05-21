#
# @file
#

project(pvpcap)

set(Boost_USE_STATIC_LIBS       OFF)                                            
set(Boost_USE_MULTITHREADED      ON)                                            
set(Boost_USE_STATIC_RUNTIME    OFF)

find_package(Boost 1.55 COMPONENTS filesystem REQUIRED)

message(STATUS "Boost include dirs: " ${Boost_INCLUDE_DIR})                      
include_directories(AFTER ${Boost_INCLUDE_DIR})

SET(PCAP_SRC
    src/libpvpcap.cpp
    src/shell.cpp
    src/ws.cpp
    src/profileformat.cpp
    src/pcap_splitter.cpp
   )

add_library (pvpcap STATIC ${PCAP_SRC})

# Set rpath
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${PVKERNEL_PLUGIN}/input-types/")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

target_link_libraries(
	pvpcap
	-lpthread
	-lpcap
	 ${Boost_LIBRARIES}
     pvkernel
)

set(INPUTS_DIST_LIST
	pvpcap
)

INSTALL(TARGETS ${INPUTS_DIST_LIST} LIBRARY DESTINATION ${PVKERNEL_PLUGIN}/input-types
				    RUNTIME DESTINATION ${PVKERNEL_PLUGIN}/input-types)
