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
include_directories(${Qt5Sql_INCLUDE_DIRS})

################################################
# Ressource declaration
################################################

set(SOURCES_DIST_LIST
source_text_file
source_perl_file
source_elasticsearch
source_splunk
)


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

# Create lib
add_library(source_database SHARED ${DB_SRC})
target_link_libraries(source_database pvkernel)

###############################################################################
# perl sources
###############################################################################
# Define ressources
SET(PERL_SRC
perl_file/PVPerlSource.cpp
perl_file/PVSourceCreatorPerlfile.cpp
perl_file/register.cpp
)

# Create lib
include_directories(AFTER ${PERL_INCLUDE_PATH})
add_library(source_perl_file SHARED ${PERL_SRC})
target_link_libraries(source_perl_file pvkernel ${PERL_LIBRARY})

###############################################################################
# Python sources
###############################################################################
if(PYTHONLIBS_FOUND)
	# AG: BEFORE here is important, or boost does not include the good
	# patchlevel.h (from Python), and does not find the good Python version !!
	include_directories(BEFORE ${PYTHON_INCLUDE_DIRS})

	# Define ressources
	SET(PYTHON_FILE_SRC
	python_file/PVPythonSource.cpp
	python_file/PVSourceCreatorPythonfile.cpp
	python_file/register.cpp
	)

	add_library(source_python_file SHARED ${PYTHON_FILE_SRC})
	target_link_libraries(source_python_file pvkernel ${PYTHON_LIBRARIES} ${Boost_LIBRARIES_PYTHON})
endif(PYTHONLIBS_FOUND)

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
# Define installation folders
###############################################################################

INSTALL(TARGETS ${SOURCES_DIST_LIST} LIBRARY DESTINATION ${PVKERNEL_PLUGIN}/sources
				     RUNTIME DESTINATION ${PVKERNEL_PLUGIN}/sources)
INSTALL(FILES ${SOURCES_DIST_FILES} DESTINATION ${SHARE_RESSOURCES}/sources)
