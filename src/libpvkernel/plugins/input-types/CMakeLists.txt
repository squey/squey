#
# @file
#

set(CMAKE_INCLUDE_CURRENT_DIR ON)

################################################
# File input plugin
################################################
# Define ressources
set(FILE_MOC_HEADERS file/PVImportFileDialog.h file/PVTextImportWidget.h)

set(FILE_SRC
file/PVInputTypeFilename.cpp
file/PVImportFileDialog.cpp
file/PVTextImportWidget.cpp
file/register.cpp
)

# Generate moc files
qt6_wrap_cpp(FILE_MOC_SRC ${FILE_MOC_HEADERS} OPTIONS)

# Generate linkage configuration
add_library(input_type_file SHARED ${FILE_SRC} ${FILE_MOC_SRC})
target_link_libraries(input_type_file PRIVATE pvkernel)

################################################
# PCAP input plugin
################################################

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/libpvkernel/plugins/input-types/")

# libraries
add_subdirectory(../common/pcap/libpvpcap pvpcap)
add_subdirectory(pcap/pcap-gui pcap-gui)

set(PCAP_SRC_FILES
pcap/PVInputTypePcap.cpp
pcap/register.cpp
)

# Set rpath
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${PVKERNEL_PLUGIN}/input-types/")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Generate linkage configuration
add_library(input_type_pcap SHARED ${PCAP_SRC_FILES})
target_link_libraries(input_type_pcap PRIVATE pvkernel
	pcap-gui-widget
	pvpcap
)

set(INPUTS_DIST_LIST
	input_type_pcap
)

install(TARGETS ${INPUTS_DIST_LIST} LIBRARY DESTINATION ${PVKERNEL_PLUGIN}/input-types
				    RUNTIME DESTINATION ${PVKERNEL_PLUGIN}/input-types)

################################################
# Database input plugin
################################################
# Declare ressource
set(DB_MOC_HEADERS
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

set(DB_LIBS Qt6::Sql)

# Generate moc and ui files
qt6_wrap_cpp(DB_MOC_SRCS ${DB_MOC_HEADERS} OPTIONS)
qt6_wrap_ui(DB_UI_SRCS ${DB_UI})

# Generate linkage configuration
add_library(input_type_database SHARED ${DB_UI_SRCS} ${DB_MOC_SRCS} ${DB_SRC_FILES})
target_link_libraries(input_type_database PRIVATE pvkernel ${DB_LIBS})


###############################################################################
# Elasticsearch
###############################################################################
# Declare ressources
set(ELASTICSEARCH_SRC_FILES
../common/elasticsearch/PVElasticsearchAPI.cpp
../common/elasticsearch/PVElasticsearchInfos.cpp
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
qt6_wrap_cpp(ELASTICSEARCH_MOC_SRCS ${ELASTICSEARCH_MOC_HEADERS} OPTIONS)
qt6_wrap_ui(ELASTICSEARCH_UI_SRCS ${ELASTICSEARCH_UI})

add_library(input_type_elasticsearch SHARED ${ELASTICSEARCH_UI_SRCS} ${ELASTICSEARCH_SRC_FILES} ${ELASTICSEARCH_MOC_SRCS})
target_link_libraries(input_type_elasticsearch PRIVATE pvkernel curl)

# ################################################
# # Parquet input plugin
# ################################################
# # Define ressources
set(PARQUET_MOC_HEADERS
parquet/PVParquetExporterWidget.h
)

set(PARQUET_SRC
../common/arrow/PVArrowRowFilter.cpp
parquet/PVParquetExporterBase.cpp
parquet/PVParquetExporter.cpp
parquet/PVParquetNrawExporter.cpp
parquet/PVParquetExporterWidget.cpp
parquet/PVInputTypeParquet.cpp
../common/parquet/PVParquetAPI.cpp
parquet/register.cpp
)

# Generate moc files
qt6_wrap_cpp(PARQUET_MOC_SRC ${PARQUET_MOC_HEADERS} OPTIONS)

# Generate linkage configuration
add_library(input_type_parquet SHARED ${PARQUET_SRC} ${PARQUET_MOC_SRC})
target_link_libraries(input_type_parquet PRIVATE pvkernel ${PARQUET_LIBS})

################################################
# JSON input plugin
################################################
# The format of the files is made by that of the Parquet files of their Arrow schema.
set(JSON_SRC
json/PVInputTypeJson.cpp
../common/arrow/PVArrowRowFilter.cpp
../common/json/PVJsonLines.cpp
../common/json/PVJsonSchema.cpp
../common/parquet/PVParquetAPI.cpp
json/register.cpp
)

add_library(input_type_json SHARED ${JSON_SRC})
target_link_libraries(input_type_json PRIVATE pvkernel ${PARQUET_LIBS})

###############################################################################
# Install plugins
###############################################################################

set(INPUTS_DIST_LIST
	input_type_file
	input_type_database
	input_type_elasticsearch
	input_type_pcap
	input_type_parquet
	input_type_json
)

install(TARGETS ${INPUTS_DIST_LIST} LIBRARY DESTINATION ${PVKERNEL_PLUGIN}/input-types
		RUNTIME DESTINATION ${PVKERNEL_PLUGIN}/input-types)
