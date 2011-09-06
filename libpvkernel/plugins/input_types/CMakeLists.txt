set(INPUTS_DIST_LIST
input_type_file
)

set(INPUTS_DIST_FILES
)

set(QT_USE_QTSQL TRUE)
include(${QT_USE_FILE})

#
# File input plugin
#
SET(FILE_MOC_HEADERS file/PVImportFileDialog.h)
QT4_WRAP_CPP(FILE_MOC_SRC ${FILE_MOC_HEADERS})

add_library(input_type_file SHARED file/PVInputTypeFilename.cpp file/PVImportFileDialog.cpp file/extract.cpp file/register.cpp ${FILE_MOC_SRC})
target_link_libraries(input_type_file ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES} ${ARCHIVE_LIBRARY})

add_library(input_type_database SHARED database/PVInputTypeDatabase.cpp ../common/database/PVDBInfos.cpp ../common/database/PVDBQuery.cpp database/register.cpp)
target_link_libraries(input_type_database ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})

if(HDFS_FOUND)
	set(INPUTS_DIST_LIST ${INPUTS_DIST_LIST} input_type_hdfs)
	include_directories(AFTER ${HDFS_INCLUDE_DIRS})
	add_library(input_type_hdfs SHARED hdfs/PVInputTypeHDFS.cpp ../common/hdfs/PVInputHDFSFile.cpp ../common/hdfs/PVInputHDFSServer.cpp hdfs/setenv.cpp hdfs/register.cpp)
	target_link_libraries(input_type_hdfs ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES} ${HDFS_LIBRARIES})
endif(HDFS_FOUND)


INSTALL(TARGETS ${INPUTS_DIST_LIST} LIBRARY DESTINATION input-types)
INSTALL(FILES ${INPUTS_DIST_FILES} DESTINATION input-types)
