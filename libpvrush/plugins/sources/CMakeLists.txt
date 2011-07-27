#
# Input type plugins 
#
include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpvcore/src/include)
include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpvrush/src/include)
include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpvfilter/src/include)

link_directories(${Picviz_Inspector_SOURCE_DIR}/libpvrush/src/ ${Picviz_Inspector_SOURCE_DIR}/libpvfilter/src/ ${Picviz_Inspector_SOURCE_DIR}/libpvcore/src/)

set(SOURCES_DIST_LIST
source_text_file
source_pcap_file
)


set(SOURCES_DIST_FILES
)

add_library(source_text_file SHARED text_file/PVSourceCreatorTextfile.cpp text_file/PVSourceCreatorTextfile.h text_file/register.cpp)
target_link_libraries(source_text_file pvrush pvfilter)

add_library(source_pcap_file SHARED pcap_file/PVSourceCreatorPcapfile.cpp pcap_file/PVSourceCreatorPcapfile.h pcap_file/register.cpp)
target_link_libraries(source_pcap_file pvrush pvfilter)

if(NOT WIN32)
	include_directories(AFTER /usr/local/include)
	link_directories(${Picviz_Inspector_SOURCE_DIR}/libpvrush/src/ ${Picviz_Inspector_SOURCE_DIR}/libpvfilter/src/ ${Picviz_Inspector_SOURCE_DIR}/libpvcore/src/ /usr/lib/jvm/java-6-sun-1.6.0.26/jre/lib/amd64/server/) 
	set(SOURCES_DIST_LIST ${SOURCES_DIST_LIST} source_text_hdfs)

	add_library(source_text_hdfs SHARED hdfs/text/PVSourceCreatorTexthdfs.cpp hdfs/text/PVSourceCreatorTexthdfs.h ../common/hdfs/PVInputHDFSFile.cpp ../common/hdfs/PVInputHDFSServer.cpp hdfs/PVInputHDFS.cpp hdfs/text/register.cpp)
	target_link_libraries(source_text_hdfs pvrush pvfilter jvm hdfs)
endif(NOT WIN32)

INSTALL(TARGETS ${SOURCES_DIST_LIST} DESTINATION sources)
INSTALL(FILES ${SOURCES_DIST_FILES} DESTINATION sources)
