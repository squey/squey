set(SOURCES_DIST_LIST
source_text_file
source_pcap_file
)


set(SOURCES_DIST_FILES
)

set(QT_USE_QTSQL TRUE)
include(${QT_USE_FILE})


add_library(source_text_file SHARED text_file/PVSourceCreatorTextfile.cpp text_file/PVSourceCreatorTextfile.h text_file/PVSourceCreatorRemoteTextfile.cpp text_file/register.cpp)
target_link_libraries(source_text_file ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES} ${PCAP_LIBRARIES})

add_library(source_pcap_file SHARED pcap_file/PVSourceCreatorPcapfile.cpp pcap_file/PVSourceCreatorPcapfile.h pcap_file/register.cpp)
target_link_libraries(source_pcap_file ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES} ${PCAP_LIBRARIES})

add_library(source_database SHARED database/PVSourceCreatorDatabase.cpp database/PVDBSource.cpp ../common/database/PVDBInfos.cpp ../common/database/PVDBQuery.cpp database/register.cpp)
target_link_libraries(source_database ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})

if(PERLLIBS_FOUND)
	include_directories(AFTER ${PERL_INCLUDE_PATH})
	add_library(source_perl_file SHARED perl_file/PVPerlSource.cpp perl_file/PVPerlSource.h perl_file/PVSourceCreatorPerlfile.cpp perl_file/PVSourceCreatorPerlfile.h perl_file/register.cpp)
	target_link_libraries(source_perl_file ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES} ${PERL_LIBRARY})
endif(PERLLIBS_FOUND)

if(HDFS_FOUND)
	add_library(source_text_hdfs SHARED hdfs/text/PVSourceCreatorTexthdfs.cpp hdfs/text/PVSourceCreatorTexthdfs.h ../common/hdfs/PVInputHDFSFile.cpp ../common/hdfs/PVInputHDFSServer.cpp hdfs/PVInputHDFS.cpp hdfs/PVChunkAlignHadoop.cpp  hdfs/PVHadoopResultServer.cpp  hdfs/PVHadoopTaskResult.cpp  hdfs/PVInputHDFS.cpp hdfs/text/register.cpp)
	target_link_libraries(source_text_hdfs ${PVKERNEL_LIBRARIES} ${HDFS_LIBRARIES})
endif(HDFS_FOUND)

INSTALL(TARGETS ${SOURCES_DIST_LIST} LIBRARY DESTINATION sources
                                     RUNTIME DESTINATION sources)
INSTALL(FILES ${SOURCES_DIST_FILES} DESTINATION sources)
