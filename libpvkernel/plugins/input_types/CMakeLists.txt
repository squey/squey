#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

set(INPUTS_DIST_LIST
input_type_file
)

if(CUSTOMER_CAPABILITY_IMPORT_REMOTE)
	set(INPUTS_DIST_LIST ${INPUTS_DIST_LIST} input_type_remotefile)
endif(CUSTOMER_CAPABILITY_IMPORT_REMOTE)

if(CUSTOMER_CAPABILITY_IMPORT_DB)
	set(INPUTS_DIST_LIST ${INPUTS_DIST_LIST} input_type_database)
endif(CUSTOMER_CAPABILITY_IMPORT_DB)

if (HDFS_FOUND)
	#if(CUSTOMER_CAPABILITY_IMPORT_HADOOP)
	#	set(INPUTS_DIST_LIST ${INPUTS_DIST_LIST} input_type_hdfs)
	#endif(CUSTOMER_CAPABILITY_IMPORT_HADOOP)
endif(HDFS_FOUND)

set(INPUTS_DIST_FILES
)

################################################
# QT5 related include and check
################################################

# For QSQLDatabase
find_package(Qt5Sql REQUIRED)
include_directories(${Qt5Sql_INCLUDE_DIRS})

# For QConcurrent
find_package(Qt5Concurrent REQUIRED)
include_directories(${Qt5Concurrent_INCLUDE_DIRS})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

################################################
# File input plugin
################################################
# Generate moc files
SET(FILE_MOC_HEADERS file/PVImportFileDialog.h)
QT5_WRAP_CPP(FILE_MOC_SRC ${FILE_MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})

# Generate linkage configuration
add_library(input_type_file SHARED file/PVInputTypeFilename.cpp file/PVImportFileDialog.cpp file/register.cpp ${FILE_MOC_SRC})
target_link_libraries(input_type_file pvkernel ${QT_LIBRARIES} ${ARCHIVE_LIBRARY})

################################################
# Database input plugin
################################################
# Declare ressource
SET(DB_MOC_HEADERS
database/PVDatabaseParamsWidget.h)

set(DB_UI
database/db_params.ui
database/db_preview.ui)

# Generate moc and ui files
QT5_WRAP_CPP(DB_MOC_SRCS ${DB_MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})
QT5_WRAP_UI(DB_UI_SRCS ${DB_UI})

# Generate linkage configuration
add_library(input_type_database SHARED ${DB_UI_SRCS} ${DB_MOC_SRCS} database/PVDatabaseParamsWidget.cpp database/PVSQLTypeMap.cpp database/PVDBPreviewWidget.cpp database/PVInputTypeDatabase.cpp database/PVDBPresets.cpp ../common/database/PVDBInfos.cpp ../common/database/PVDBQuery.cpp database/register.cpp)
target_link_libraries(input_type_database pvkernel ${QT_LIBRARIES})

#
# HDFS input plugin
#
if(HDFS_FOUND)
#	include_directories(AFTER ${HDFS_INCLUDE_DIRS})
#	SET(HDFS_UI hdfs/PVImportHDFSDlg.ui)
#	SET(HDFS_MOC_HEADERS hdfs/PVImportHDFSDlg.h)
#	QT5_WRAP_CPP(HDFS_MOC_SRCS ${HDFS_MOC_HEADERS})
#	QT5_WRAP_UI(HDFS_UI_SRCS ${HDFS_UI})
#	add_library(input_type_hdfs SHARED hdfs/PVImportHDFSDlg.cpp hdfs/PVInputTypeHDFS.cpp ../common/hdfs/PVInputHDFSFile.cpp ../common/hdfs/PVInputHDFSServer.cpp hdfs/setenv.cpp hdfs/register.cpp ${HDFS_MOC_SRCS} ${HDFS_UI_SRCS})
#	target_link_libraries(input_type_hdfs ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES} ${HDFS_LIBRARIES})
endif(HDFS_FOUND)

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
)

# Generate moc files
QT5_WRAP_CPP(REMOTEFILE_MOC_SRCS ${REMOTEFILE_MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})

# Generate linkage configuration
include_directories(AFTER ${CURL_INCLUDE_DIR})
add_library(input_type_remotefile SHARED remote_file/addmachinedialog.cpp remote_file/fileconnectiondialog.cpp remote_file/filedownloader.cpp remote_file/logviewerwidget.cpp remote_file/PVLogViewerDialog.cpp remote_file/PVInputTypeRemoteFilename.cpp remote_file/register.cpp ${REMOTEFILE_MOC_SRCS} file/PVInputTypeFilename.cpp)
target_link_libraries(input_type_remotefile pvkernel ${QT_LIBRARIES} ${ARCHIVE_LIBRARY} ${CURL_LIBRARY})

################################################
# Arcsight
################################################
# Declare ressource
set(ARCSIGHT_MOC_HEADERS
arcsight/PVArcsightParamsWidget.h
#arcsight/PVArcsightPreviewWidget.h
)

set(ARCSIGHT_UI
arcsight/arcsight_params.ui)

# Generate moc and ui files
QT5_WRAP_CPP(ARCSIGHT_MOC_SRCS ${ARCSIGHT_MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})
QT5_WRAP_UI(ARCSIGHT_UI_SRCS ${ARCSIGHT_UI})

# Generate linkage configuration
add_library(input_type_arcsight SHARED arcsight/PVArcsightParamsWidget.cpp arcsight/PVInputTypeArcsight.cpp arcsight/register.cpp ${ARCSIGHT_MOC_SRCS} ../common/arcsight/PVArcsightInfos.cpp ../common/arcsight/PVArcsightQuery.cpp ${ARCSIGHT_UI_SRCS})
target_link_libraries(input_type_arcsight pvkernel ${QT_LIBRARIES} /usr/local/lib/libarcsight.a gsoap++)

INSTALL(TARGETS ${INPUTS_DIST_LIST} LIBRARY DESTINATION input-types
                                    RUNTIME DESTINATION input-types)
INSTALL(FILES ${INPUTS_DIST_FILES} DESTINATION input-types)
