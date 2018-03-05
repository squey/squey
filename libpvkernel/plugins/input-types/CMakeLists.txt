#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

################################################
# QT5 related include and check
################################################

# For QSQLDatabase
find_package(Qt5Sql REQUIRED)
include_directories(SYSTEM ${Qt5Sql_INCLUDE_DIRS})

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
QT5_WRAP_CPP(FILE_MOC_SRC ${FILE_MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})

# Generate linkage configuration
add_library(input_type_file SHARED ${FILE_SRC} ${FILE_MOC_SRC})
target_link_libraries(input_type_file pvkernel ${ARCHIVE_LIBRARY})

################################################
# PCAP input plugin
################################################

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/libpvkernel/plugins/input-types/")

# LightPcapNg
include_directories(SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../common/pcap/LightPcapNg/include)
cmake_minimum_required(VERSION 2.8)
include(ExternalProject)
SET(LIGHTPCAPNG_LIBRARY_DIR ${CMAKE_CURRENT_LIST_DIR}/../common/pcap/LightPcapNg)
SET(LIGHTPCAPNG_LIBRARY ${LIGHTPCAPNG_LIBRARY_DIR}/liblight_pcapng_static.a)
ExternalProject_Add(LightPcapNg
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../common/pcap/LightPcapNg
    BUILD_IN_SOURCE 1
    STEP_TARGETS build
  	EXCLUDE_FROM_ALL TRUE
  	BUILD_BYPRODUCTS ${LIGHTPCAPNG_LIBRARY}
  	INSTALL_COMMAND cmake -E echo "Skipping install step."
)

include_directories("../common/pcap/libpvpcap/include/")
include_directories("pcap/pcap-gui/")

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
QT5_WRAP_CPP(PCAP_MOC_SRC ${PCAP_MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})

# Generate linkage configuration
add_library(input_type_pcap SHARED ${PCAP_SRC_FILES} ${PCAP_MOC_SRC})
add_dependencies(pvpcap LightPcapNg)
target_link_libraries(
	input_type_pcap
	pvkernel
	${ARCHIVE_LIBRARY}
	pcap-gui-widget
	pvpcap
	${LIGHTPCAPNG_LIBRARY}
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

set(DB_LIBS Qt5Sql)

# Generate moc and ui files
QT5_WRAP_CPP(DB_MOC_SRCS ${DB_MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})
QT5_WRAP_UI(DB_UI_SRCS ${DB_UI})

# Generate linkage configuration
add_library(input_type_database SHARED ${DB_UI_SRCS} ${DB_MOC_SRCS} ${DB_SRC_FILES})
target_link_libraries(input_type_database pvkernel ${DB_LIBS})

################################################
# Remote file input plugin
################################################
# Declare ressource
set(REMOTEFILE_MOC_HEADERS
remote_file/include/addmachinedialog.h
remote_file/include/fileconnectiondialog.h
remote_file/include/filedownloader.h
remote_file/include/logviewerwidget.h
remote_file/include/PVLogViewerDialog.h
file/PVImportFileDialog.h
)

SET(REMOTEFILE_SRC
remote_file/addmachinedialog.cpp
remote_file/fileconnectiondialog.cpp
remote_file/filedownloader.cpp
remote_file/logviewerwidget.cpp
remote_file/PVLogViewerDialog.cpp
remote_file/PVInputTypeRemoteFilename.cpp
remote_file/register.cpp
file/PVInputTypeFilename.cpp
file/PVImportFileDialog.cpp
)

# Generate moc files
QT5_WRAP_CPP(REMOTEFILE_MOC_SRCS ${REMOTEFILE_MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})

# Generate linkage configuration
include_directories(SYSTEM AFTER ${CURL_INCLUDE_DIR})
add_library(input_type_remotefile SHARED ${REMOTEFILE_SRC} ${REMOTEFILE_MOC_SRCS})
target_link_libraries(input_type_remotefile pvkernel ${ARCHIVE_LIBRARY} ${CURL_LIBRARIES})

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
QT5_WRAP_CPP(ELASTICSEARCH_MOC_SRCS ${ELASTICSEARCH_MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})
QT5_WRAP_UI(ELASTICSEARCH_UI_SRCS ${ELASTICSEARCH_UI})

add_library(input_type_elasticsearch SHARED ${ELASTICSEARCH_UI_SRCS} ${ELASTICSEARCH_SRC_FILES} ${ELASTICSEARCH_MOC_SRCS})
target_link_libraries(input_type_elasticsearch pvkernel curl)

###############################################################################
# Splunk
###############################################################################
# Declare ressources
set(SPLUNK_SRC_FILES
../common/splunk/PVSplunkAPI.cpp
../common/splunk/PVSplunkJsonConverter.cpp
../common/splunk/PVSplunkInfos.cpp
../common/splunk/PVSplunkQuery.cpp
common/PVParamsWidget.cpp
splunk/PVSplunkParamsWidget.cpp
splunk/PVSplunkPresets.cpp
splunk/PVInputTypeSplunk.cpp
splunk/register.cpp
)

set(SPLUNK_MOC_HEADERS
common/PVParamsWidget.h
splunk/PVSplunkParamsWidget.h
)

set(SPLUNK_UI
common/widget_params.ui)

# Generate Qt Specific files
QT5_WRAP_CPP(SPLUNK_MOC_SRCS ${SPLUNK_MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})
QT5_WRAP_UI(SPLUNK_UI_SRCS ${SPLUNK_UI})

add_library(input_type_splunk SHARED ${SPLUNK_UI_SRCS} ${SPLUNK_SRC_FILES} ${SPLUNK_MOC_SRCS})
target_link_libraries(input_type_splunk pvkernel curl)

###############################################################################
# Install plugins
###############################################################################

set(INPUTS_DIST_LIST
	input_type_file
	input_type_remotefile
	input_type_database
	input_type_elasticsearch
	input_type_splunk
	)

INSTALL(TARGETS ${INPUTS_DIST_LIST} LIBRARY DESTINATION ${PVKERNEL_PLUGIN}/input-types
				    RUNTIME DESTINATION ${PVKERNEL_PLUGIN}/input-types)
