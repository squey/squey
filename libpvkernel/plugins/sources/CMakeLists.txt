#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2015

################################################
# QT5 related include and check
################################################

# For QSQLQuery
find_package(Qt5Sql REQUIRED)
include_directories(${Qt5Sql_INCLUDE_DIRS})

################################################
# Ressource declaration
################################################

set(SOURCES_DIST_LIST
source_text_file
source_pcap_file
source_perl_file
source_elasticsearch
source_splunk
#source_arcsight
)


set(SOURCES_DIST_FILES
)

################################################
# Declare the new library and linkage options
################################################

add_library(source_text_file SHARED text_file/PVSourceCreatorTextfile.cpp text_file/PVSourceCreatorTextfile.h text_file/PVSourceCreatorRemoteTextfile.cpp text_file/register.cpp)
target_link_libraries(source_text_file pvkernel ${PCAP_LIBRARIES})

add_library(source_pcap_file SHARED pcap_file/PVSourceCreatorPcapfile.cpp pcap_file/PVSourceCreatorPcapfile.h pcap_file/register.cpp)
target_link_libraries(source_pcap_file pvkernel ${PCAP_LIBRARIES})

add_library(source_database SHARED database/PVSourceCreatorDatabase.cpp database/PVDBSource.cpp ../common/database/PVDBInfos.cpp ../common/database/PVDBQuery.cpp database/register.cpp)
target_link_libraries(source_database pvkernel)

if(PERLLIBS_FOUND)
	include_directories(AFTER ${PERL_INCLUDE_PATH})
	add_library(source_perl_file SHARED perl_file/PVPerlSource.cpp perl_file/PVPerlSource.h perl_file/PVSourceCreatorPerlfile.cpp perl_file/PVSourceCreatorPerlfile.h perl_file/register.cpp)
	target_link_libraries(source_perl_file pvkernel ${PERL_LIBRARY})


	#set(SOURCES_DIST_LIST ${SOURCES_DIST_LIST} source_perl_file)
endif(PERLLIBS_FOUND)

if(PYTHONLIBS_FOUND)
	# AG: BEFORE here is important, or boost does not include the good patchlevel.h (from Python), and does not
	# find the good Python version !!
	include_directories(BEFORE ${PYTHON_INCLUDE_DIRS})
	add_library(source_python_file SHARED python_file/PVPythonSource.cpp python_file/PVPythonSource.h python_file/PVSourceCreatorPythonfile.cpp python_file/PVSourceCreatorPythonfile.h python_file/register.cpp)
	target_link_libraries(source_python_file pvkernel ${PYTHON_LIBRARIES} ${Boost_LIBRARIES_PYTHON})
endif(PYTHONLIBS_FOUND)


if(HDFS_FOUND)
#	add_library(source_text_hdfs SHARED hdfs/text/PVSourceCreatorTexthdfs.cpp hdfs/text/PVSourceCreatorTexthdfs.h ../common/hdfs/PVInputHDFSFile.cpp ../common/hdfs/PVInputHDFSServer.cpp hdfs/PVInputHDFS.cpp hdfs/PVChunkAlignHadoop.cpp  hdfs/PVHadoopResultServer.cpp  hdfs/PVHadoopTaskResult.cpp  hdfs/PVInputHDFS.cpp hdfs/text/register.cpp)
#	target_link_libraries(source_text_hdfs pvkernel ${HDFS_LIBRARIES})
endif(HDFS_FOUND)

# Arcsight
add_library(source_arcsight SHARED arcsight/PVArcsightSource.cpp arcsight/PVSourceCreatorArcsight.cpp arcsight/register.cpp ../common/arcsight/PVArcsightInfos.cpp ../common/arcsight/PVArcsightQuery.cpp)
target_link_libraries(source_arcsight /usr/local/lib/libarcsight.a pvkernel gsoap++)

###############################################################################
# Elasticsearch
###############################################################################

set(ELASTICSEARCH_SRC_FILES
elasticsearch/PVElasticsearchSource.cpp
elasticsearch/PVSourceCreatorElasticsearch.cpp
../common/elasticsearch/PVElasticsearchInfos.cpp
../common/elasticsearch/PVElasticsearchQuery.cpp
../common/elasticsearch/PVElasticsearchAPI.cpp
../common/elasticsearch/PVElasticsearchJsonConverter.cpp
elasticsearch/register.cpp
)

add_library(source_elasticsearch SHARED ${ELASTICSEARCH_SRC_FILES})
target_link_libraries(source_elasticsearch curl)

###############################################################################
# Splunk
###############################################################################

set(SPLUNK_SRC_FILES
splunk/PVSplunkSource.cpp
splunk/PVSourceCreatorSplunk.cpp
../common/splunk/PVSplunkInfos.cpp
../common/splunk/PVSplunkQuery.cpp
../common/splunk/PVSplunkAPI.cpp
../common/splunk/PVSplunkJsonConverter.cpp
splunk/register.cpp
)

add_library(source_splunk SHARED ${SPLUNK_SRC_FILES})
target_link_libraries(source_splunk curl)

###############################################################################

INSTALL(TARGETS ${SOURCES_DIST_LIST} LIBRARY DESTINATION sources
                                     RUNTIME DESTINATION sources)
INSTALL(FILES ${SOURCES_DIST_FILES} DESTINATION sources)
