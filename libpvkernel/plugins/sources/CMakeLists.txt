#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

################################################
# QT5 related include and check
################################################

# For QSQLQuery
find_package(Qt5Sql REQUIRED)
include_directories(SYSTEM ${Qt5Sql_INCLUDE_DIRS})

################################################
# Ressource declaration
################################################

set(SOURCES_DIST_FILES
)

################################################
# Declare the new library and linkage options
################################################

###############################################################################
# text_file sources
###############################################################################
# Define ressources
SET(TEXT_FILE_SRC
text_file/PVSourceCreatorTextfile.cpp
text_file/PVSourceCreatorRemoteTextfile.cpp
text_file/register.cpp
)

# Create lib
add_library(source_text_file SHARED ${TEXT_FILE_SRC})
target_link_libraries(source_text_file pvkernel)

###############################################################################
# database sources
###############################################################################
# Define ressources
SET(DB_SRC
database/PVSourceCreatorDatabase.cpp
database/PVDBSource.cpp
../common/database/PVDBInfos.cpp
../common/database/PVDBQuery.cpp
database/register.cpp
)

# For QSQLDatabase
find_package(Qt5Sql REQUIRED)
include_directories(SYSTEM ${Qt5Sql_INCLUDE_DIRS})

# Create lib
add_library(source_database SHARED ${DB_SRC})
target_link_libraries(source_database pvkernel Qt5Sql)

###############################################################################
# Elasticsearch
###############################################################################
# Define ressources
set(ELASTICSEARCH_SRC_FILES
elasticsearch/PVElasticsearchSource.cpp
elasticsearch/PVSourceCreatorElasticsearch.cpp
../common/elasticsearch/PVElasticsearchInfos.cpp
../common/elasticsearch/PVElasticsearchQuery.cpp
../common/elasticsearch/PVElasticsearchAPI.cpp
../common/elasticsearch/PVElasticsearchJsonConverter.cpp
elasticsearch/register.cpp
)

# Create lib
add_library(source_elasticsearch SHARED ${ELASTICSEARCH_SRC_FILES})
target_link_libraries(source_elasticsearch curl)

###############################################################################
# Splunk
###############################################################################
# Define ressources
set(SPLUNK_SRC_FILES
splunk/PVSplunkSource.cpp
splunk/PVSourceCreatorSplunk.cpp
../common/splunk/PVSplunkInfos.cpp
../common/splunk/PVSplunkQuery.cpp
../common/splunk/PVSplunkAPI.cpp
../common/splunk/PVSplunkJsonConverter.cpp
splunk/register.cpp
)

# Create lib
add_library(source_splunk SHARED ${SPLUNK_SRC_FILES})
target_link_libraries(source_splunk curl)

###############################################################################
# pcap sources
###############################################################################

# Define ressources
set(PCAP_SRC_FILES
	pcap/PVSourceCreatorPcap.cpp
	pcap/register.cpp
)

# Create lib
add_library(source_pcap SHARED ${PCAP_SRC_FILES})
target_link_libraries(source_pcap pvkernel)

###############################################################################
# ERF sources
###############################################################################

# Define ressources
set(ERF_SRC_FILES
	../common/erf/PVERFAPI.cpp
	erf/PVERFSource.cpp
	erf/PVSourceCreatorERF.cpp
	erf/register.cpp
)

set(ERF_LIBS
    /app/lib/libhdf5_cpp.so
	/app/lib/liberf_cpp_sp.so
)

# Create lib
add_library(source_erf SHARED ${ERF_SRC_FILES})
target_link_libraries(source_erf pvkernel ${ERF_LIBS})

###############################################################################
# Define installation folders
###############################################################################
set(SOURCES_DIST_LIST
source_text_file
source_elasticsearch
source_splunk
source_pcap
source_erf
)
INSTALL(TARGETS ${SOURCES_DIST_LIST} LIBRARY DESTINATION ${PVKERNEL_PLUGIN}/sources
				     RUNTIME DESTINATION ${PVKERNEL_PLUGIN}/sources)

###############################################################################
# Define installation folders
###############################################################################

INSTALL(TARGETS ${SOURCES_DIST_LIST} LIBRARY DESTINATION ${PVKERNEL_PLUGIN}/sources
				     RUNTIME DESTINATION ${PVKERNEL_PLUGIN}/sources)
INSTALL(FILES ${SOURCES_DIST_FILES} DESTINATION ${SHARE_RESSOURCES}/sources)
