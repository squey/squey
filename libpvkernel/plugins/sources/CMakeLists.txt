set(SOURCES_DIST_LIST
source_text_file
source_pcap_file
)


set(SOURCES_DIST_FILES
)

set(QT_USE_QTSQL TRUE)
include(${QT_USE_FILE})


add_library(source_text_file SHARED text_file/PVSourceCreatorTextfile.cpp text_file/PVSourceCreatorTextfile.h text_file/register.cpp)
target_link_libraries(source_text_file ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES} ${PCAP_LIBRARIES})

add_library(source_pcap_file SHARED pcap_file/PVSourceCreatorPcapfile.cpp pcap_file/PVSourceCreatorPcapfile.h pcap_file/register.cpp)
target_link_libraries(source_pcap_file ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES} ${PCAP_LIBRARIES})

add_library(source_database SHARED database/PVSourceCreatorDatabase.cpp database/PVDBSource.cpp ../common/database/PVDBInfos.cpp ../common/database/PVDBQuery.cpp database/register.cpp)
target_link_libraries(source_database ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})

if(HDFS_FOUND)
	include_directories(AFTER ${HDFS_INCLUDE_DIRS})
	set(SOURCES_DIST_LIST ${SOURCES_DIST_LIST} source_text_hdfs)

	add_library(source_text_hdfs SHARED hdfs/text/PVSourceCreatorTexthdfs.cpp hdfs/text/PVSourceCreatorTexthdfs.h ../common/hdfs/PVInputHDFSFile.cpp ../common/hdfs/PVInputHDFSServer.cpp hdfs/PVInputHDFS.cpp hdfs/PVChunkAlignHadoop.cpp  hdfs/PVHadoopResultServer.cpp  hdfs/PVHadoopTaskResult.cpp  hdfs/PVInputHDFS.cpp hdfs/text/register.cpp)
	target_link_libraries(source_text_hdfs ${PVKERNEL_LIBRARIES} ${HDFS_LIBRARIES})
endif(HDFS_FOUND)

INSTALL(TARGETS ${SOURCES_DIST_LIST} LIBRARY DESTINATION sources)
INSTALL(FILES ${SOURCES_DIST_FILES} DESTINATION sources)
