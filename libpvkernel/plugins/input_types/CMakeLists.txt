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
	if(CUSTOMER_CAPABILITY_IMPORT_HADOOP)
		set(INPUTS_DIST_LIST ${INPUTS_DIST_LIST} input_type_hdfs)
	endif(CUSTOMER_CAPABILITY_IMPORT_HADOOP)
endif(HDFS_FOUND)

set(INPUTS_DIST_FILES
)

set(QT_USE_QTSQL TRUE)
include(${QT_USE_FILE})

#
# File input plugin
#
SET(FILE_MOC_HEADERS file/PVImportFileDialog.h)
QT4_WRAP_CPP(FILE_MOC_SRC ${FILE_MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})

add_library(input_type_file SHARED file/PVInputTypeFilename.cpp file/PVImportFileDialog.cpp file/register.cpp ${FILE_MOC_SRC})
target_link_libraries(input_type_file ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES} ${ARCHIVE_LIBRARY})

#
# Database input plugin
#
SET(DB_MOC_HEADERS database/PVDatabaseParamsWidget.h)
QT4_WRAP_CPP(DB_MOC_SRCS ${DB_MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})
set(DB_UI database/db_params.ui database/db_preview.ui)
QT4_WRAP_UI(DB_UI_SRCS ${DB_UI})

add_library(input_type_database SHARED ${DB_UI_SRCS} ${DB_MOC_SRCS} database/PVDatabaseParamsWidget.cpp database/PVSQLTypeMap.cpp database/PVDBPreviewWidget.cpp database/PVInputTypeDatabase.cpp database/PVDBPresets.cpp ../common/database/PVDBInfos.cpp ../common/database/PVDBQuery.cpp database/register.cpp)
target_link_libraries(input_type_database ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})

#
# HDFS input plugin
#
if(HDFS_FOUND)
	include_directories(AFTER ${HDFS_INCLUDE_DIRS})
	add_library(input_type_hdfs SHARED hdfs/PVInputTypeHDFS.cpp ../common/hdfs/PVInputHDFSFile.cpp ../common/hdfs/PVInputHDFSServer.cpp hdfs/setenv.cpp hdfs/register.cpp)
	target_link_libraries(input_type_hdfs ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES} ${HDFS_LIBRARIES})
endif(HDFS_FOUND)

#
# Remote file input plugin
#
set(REMOTEFILE_MOC_HEADERS remote_file/include/addmachinedialog.h remote_file/include/connectionsettings.h remote_file/include/fileconnectiondialog.h remote_file/include/filedownloader.h remote_file/include/logviewer_config.h remote_file/include/logviewer_export.h remote_file/include/logviewerwidget.h remote_file/include/logviewerwidget_p.h remote_file/include/PVLogViewerDialog.h)
QT4_WRAP_CPP(REMOTEFILE_MOC_SRCS ${REMOTEFILE_MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})
include_directories(AFTER ${CURL_INCLUDE_DIR})
add_library(input_type_remotefile SHARED remote_file/addmachinedialog.cpp remote_file/fileconnectiondialog.cpp remote_file/filedownloader.cpp remote_file/logviewerwidget.cpp remote_file/PVLogViewerDialog.cpp remote_file/PVInputTypeRemoteFilename.cpp remote_file/register.cpp ${REMOTEFILE_MOC_SRCS} file/PVInputTypeFilename.cpp)
target_link_libraries(input_type_remotefile ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES} ${ARCHIVE_LIBRARY} ${CURL_LIBRARY})


INSTALL(TARGETS ${INPUTS_DIST_LIST} LIBRARY DESTINATION input-types
                                    RUNTIME DESTINATION input-types)
INSTALL(FILES ${INPUTS_DIST_FILES} DESTINATION input-types)
