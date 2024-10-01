#
# @file
#

################################################
# QT6 related include and check
################################################

# For QSQLDatabase
find_package(Qt6Sql REQUIRED)
include_directories(SYSTEM ${Qt6Sql_INCLUDE_DIRS})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

################################################
# File input plugin
################################################
# Define ressources
SET(FILE_MOC_HEADERS file/PVImportFileDialog.h)

SET(FILE_SRC
file/PVInputTypeFilename.cpp
file/PVImportFileDialog.cpp
file/register.cpp
)

# Generate moc files
QT6_WRAP_CPP(FILE_MOC_SRC ${FILE_MOC_HEADERS} OPTIONS ${QT6_WRAP_OPTIONS})

# Generate linkage configuration
add_library(input_type_file SHARED ${FILE_SRC} ${FILE_MOC_SRC})
target_link_libraries(input_type_file pvkernel ${ARCHIVE_LIBRARY})

################################################
# PCAP input plugin
################################################

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/libpvkernel/plugins/input-types/")

include_directories(SYSTEM "../common/pcap/libpvpcap/include/")
include_directories(SYSTEM "pcap/pcap-gui/")

# libraries
add_subdirectory(../common/pcap/libpvpcap pvpcap)
add_subdirectory(pcap/pcap-gui pcap-gui)

# Define ressources
SET(PCAP_MOC_HEADERS 
pcap/PVPcapParamsWidget.h
)

SET(PCAP_SRC_FILES
pcap/PVInputTypePcap.cpp
pcap/PVPcapParamsWidget.cpp
pcap/register.cpp
)

# Set rpath
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${PVKERNEL_PLUGIN}/input-types/")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Generate moc files
QT6_WRAP_CPP(PCAP_MOC_SRC ${PCAP_MOC_HEADERS} OPTIONS ${QT6_WRAP_OPTIONS})

# Generate linkage configuration
add_library(input_type_pcap SHARED ${PCAP_SRC_FILES} ${PCAP_MOC_SRC})
target_link_libraries(
	input_type_pcap
	pvkernel
	${ARCHIVE_LIBRARY}
	pcap-gui-widget
	pvpcap
	light_pcapng_static
)

set(INPUTS_DIST_LIST
	input_type_pcap
)

INSTALL(TARGETS ${INPUTS_DIST_LIST} LIBRARY DESTINATION ${PVKERNEL_PLUGIN}/input-types
				    RUNTIME DESTINATION ${PVKERNEL_PLUGIN}/input-types)

################################################
# Database input plugin
################################################
# Declare ressource
SET(DB_MOC_HEADERS
database/PVDatabaseParamsWidget.h)

set(DB_SRC_FILES
../common/database/PVDBInfos.cpp
../common/database/PVDBQuery.cpp
../common/database/PVSQLTypeMap.cpp
database/PVDBPresets.cpp
database/PVDBPreviewWidget.cpp
database/PVDatabaseParamsWidget.cpp
database/PVInputTypeDatabase.cpp
database/register.cpp)

set(DB_UI
database/db_params.ui
database/db_preview.ui)

set(DB_LIBS Qt6Sql)

# Generate moc and ui files
QT6_WRAP_CPP(DB_MOC_SRCS ${DB_MOC_HEADERS} OPTIONS ${QT6_WRAP_OPTIONS})
QT6_WRAP_UI(DB_UI_SRCS ${DB_UI})

# Generate linkage configuration
add_library(input_type_database SHARED ${DB_UI_SRCS} ${DB_MOC_SRCS} ${DB_SRC_FILES})
target_link_libraries(input_type_database pvkernel ${DB_LIBS})


###############################################################################
# Elasticsearch
###############################################################################
# Declare ressources
set(ELASTICSEARCH_SRC_FILES
../common/elasticsearch/PVElasticsearchAPI.cpp
../common/elasticsearch/PVElasticsearchInfos.cpp
../common/elasticsearch/PVElasticsearchJsonConverter.cpp
../common/elasticsearch/PVElasticsearchQuery.cpp
common/PVParamsWidget.cpp
elasticsearch/PVElasticsearchParamsWidget.cpp
elasticsearch/PVElasticsearchPresets.cpp
elasticsearch/PVInputTypeElasticsearch.cpp
elasticsearch/register.cpp
)

set(ELASTICSEARCH_MOC_HEADERS
common/PVParamsWidget.h
elasticsearch/PVElasticsearchParamsWidget.h
)

set(ELASTICSEARCH_UI
common/widget_params.ui)

# Generate Qt Specific files
QT6_WRAP_CPP(ELASTICSEARCH_MOC_SRCS ${ELASTICSEARCH_MOC_HEADERS} OPTIONS ${QT6_WRAP_OPTIONS})
QT6_WRAP_UI(ELASTICSEARCH_UI_SRCS ${ELASTICSEARCH_UI})

add_library(input_type_elasticsearch SHARED ${ELASTICSEARCH_UI_SRCS} ${ELASTICSEARCH_SRC_FILES} ${ELASTICSEARCH_MOC_SRCS})
target_link_libraries(input_type_elasticsearch pvkernel curl)

################################################
# Parquet input plugin
################################################
# Define ressources
SET(PARQUET_MOC_HEADERS
parquet/PVParquetExporterWidget.h
)

SET(PARQUET_SRC
parquet/PVParquetExporter.cpp
parquet/PVParquetExporterWidget.cpp
parquet/PVInputTypeParquet.cpp
../common/parquet/PVParquetAPI.cpp
parquet/register.cpp
)

# Generate moc files
QT6_WRAP_CPP(PARQUET_MOC_SRC ${PARQUET_MOC_HEADERS} OPTIONS ${QT6_WRAP_OPTIONS})

# Generate linkage configuration
add_library(input_type_parquet SHARED ${PARQUET_SRC} ${PARQUET_MOC_SRC})
target_link_libraries(input_type_parquet pvkernel ${ARCHIVE_LIBRARY})

###############################################################################
# Install plugins
###############################################################################

set(INPUTS_DIST_LIST
	input_type_file
	input_type_database
	input_type_elasticsearch
	input_type_parquet
	)

INSTALL(TARGETS ${INPUTS_DIST_LIST} LIBRARY DESTINATION ${PVKERNEL_PLUGIN}/input-types
				    RUNTIME DESTINATION ${PVKERNEL_PLUGIN}/input-types)
