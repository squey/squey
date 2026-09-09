#
# @file
#

project(pvpcap)

set(Boost_USE_STATIC_LIBS       OFF)                                            
set(Boost_USE_MULTITHREADED      ON)                                            
set(Boost_USE_STATIC_RUNTIME    OFF)

find_package(Boost 1.55 COMPONENTS filesystem REQUIRED)

set(PCAP_SRC
    src/libpvpcap.cpp
    src/shell.cpp
    src/ws.cpp
    src/profileformat.cpp
    src/pcap_splitter.cpp
   )

add_library(pvpcap STATIC ${PCAP_SRC})
# SYSTEM, as the directory-wide include this replaces already was: sniff_def.h
# redefines TH_FLAGS, which the macOS SDK also defines, and the diagnostic is of
# no use on vendored headers.
target_include_directories(pvpcap SYSTEM PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Set rpath
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${PVKERNEL_PLUGIN}/input-types/")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_package(Threads REQUIRED)
target_link_libraries(pvpcap
	PUBLIC pvkernel ${Boost_LIBRARIES}
	PRIVATE pcap Threads::Threads)

set(INPUTS_DIST_LIST
	pvpcap
)

install(TARGETS ${INPUTS_DIST_LIST} LIBRARY DESTINATION ${PVKERNEL_PLUGIN}/input-types
				    RUNTIME DESTINATION ${PVKERNEL_PLUGIN}/input-types)
