#
# @file
#

################################################
# QT6 related include and check
################################################

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
set(TEXT_FILE_SRC
text_file/PVSourceCreatorTextfile.cpp
text_file/PVSourceCreatorRemoteTextfile.cpp
text_file/register.cpp
)

# Create lib
add_library(source_text_file SHARED ${TEXT_FILE_SRC})
target_link_libraries(source_text_file PRIVATE pvkernel)

###############################################################################
# database sources
###############################################################################
# Define ressources
set(DB_SRC
database/PVSourceCreatorDatabase.cpp
database/PVDBSource.cpp
../common/database/PVSQLTypeMap.cpp
../common/database/PVDBInfos.cpp
../common/database/PVDBQuery.cpp
database/register.cpp
)

# Create lib
add_library(source_database SHARED ${DB_SRC})
target_link_libraries(source_database PRIVATE pvkernel Qt6::Sql)

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
elasticsearch/register.cpp
)

# Create lib
add_library(source_elasticsearch SHARED ${ELASTICSEARCH_SRC_FILES})
target_link_libraries(source_elasticsearch PRIVATE pvkernel curl ${TBB_LIBRARIES})

###############################################################################
# pcap sources
###############################################################################

# # Define ressources
set(PCAP_SRC_FILES
	pcap/PVSourceCreatorPcap.cpp
	pcap/register.cpp
)

# Create lib
add_library(source_pcap SHARED ${PCAP_SRC_FILES})
target_link_libraries(source_pcap PRIVATE pvkernel pvpcap)

###############################################################################
# parquet sources
###############################################################################

# # Define ressources
set(PARQUET_SRC_FILES
	../common/arrow/PVArrowRowFilter.cpp
	../common/parquet/PVParquetAPI.cpp
	parquet/PVParquetSource.cpp
	parquet/PVParquetBinaryChunk.cpp
	parquet/PVSourceCreatorParquet.cpp
	parquet/register.cpp
)

# Create lib
add_library(source_parquet SHARED ${PARQUET_SRC_FILES})
target_link_libraries(source_parquet PRIVATE pvkernel Arrow::arrow_shared Parquet::parquet_shared thrift re2::re2 snappy)

###############################################################################
# json sources
###############################################################################

# The rows are read as Arrow arrays, which become those of the NRAW as Parquet rows do.
set(JSON_SRC_FILES
	../common/arrow/PVArrowRowFilter.cpp
	../common/json/PVJsonLines.cpp
	../common/json/PVJsonRows.cpp
	../common/json/PVJsonSchema.cpp
	../common/parquet/PVParquetAPI.cpp
	parquet/PVParquetBinaryChunk.cpp
	json/PVJsonSource.cpp
	json/PVSourceCreatorJson.cpp
	json/register.cpp
)

add_library(source_json SHARED ${JSON_SRC_FILES})
target_link_libraries(source_json PRIVATE pvkernel ${PARQUET_LIBS})

###############################################################################
# Define installation folders
###############################################################################
set(SOURCES_DIST_LIST
source_text_file
source_database
source_elasticsearch
source_pcap
source_parquet
source_json
)
install(TARGETS ${SOURCES_DIST_LIST} LIBRARY DESTINATION ${PVKERNEL_PLUGIN}/sources
		RUNTIME DESTINATION ${PVKERNEL_PLUGIN}/sources)

###############################################################################
# Define installation folders
###############################################################################

install(TARGETS ${SOURCES_DIST_LIST} LIBRARY DESTINATION ${PVKERNEL_PLUGIN}/sources
		RUNTIME DESTINATION ${PVKERNEL_PLUGIN}/sources)
install(FILES ${SOURCES_DIST_FILES} DESTINATION ${SHARE_RESSOURCES}/sources)
