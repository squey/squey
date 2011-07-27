#
# Input type plugins 
#
include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpvcore/src/include)
include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpvrush/src/include)
include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpvfilter/src/include)

link_directories(${Picviz_Inspector_SOURCE_DIR}/libpvrush/src/ ${Picviz_Inspector_SOURCE_DIR}/libpvfilter/src/ ${Picviz_Inspector_SOURCE_DIR}/libpvcore/src/)

set(INPUTS_DIST_LIST
input_type_file
)

set(INPUTS_DIST_FILES
)

#
# File input plugin
#

SET(FILE_MOC_HEADERS file/PVImportFileDialog.h)
QT4_WRAP_CPP(FILE_MOC_SRC ${FILE_MOC_HEADERS})

add_library(input_type_file SHARED file/PVInputTypeFilename.cpp file/PVImportFileDialog.cpp file/extract.cpp file/register.cpp ${FILE_MOC_SRC})
target_link_libraries(input_type_file pvrush pvfilter ${QT_LIBRARIES})

#if(NOT WIN32)
#	set(INPUTS_DIST_LIST ${INPUTS_DIST_LIST} input_type_hdfs)
#	include_directories(AFTER /usr/local/include)
#	link_directories(${Picviz_Inspector_SOURCE_DIR}/libpvrush/src/ ${Picviz_Inspector_SOURCE_DIR}/libpvfilter/src/ ${Picviz_Inspector_SOURCE_DIR}/libpvcore/src/ /usr/lib/jvm/java-6-sun-1.6.0.26/jre/lib/amd64/server/) 
#	add_library(input_type_hdfs SHARED hdfs/PVInputTypeHDFS.cpp ../common/hdfs/PVInputHDFSFile.cpp ../common/hdfs/PVInputHDFSServer.cpp hdfs/setenv.cpp hdfs/register.cpp ${FILE_MOC_SRC})
#	target_link_libraries(input_type_hdfs pvrush pvfilter ${QT_LIBRARIES} jvm hdfs)
#endif(NOT WIN32)


INSTALL(TARGETS ${INPUTS_DIST_LIST} DESTINATION input-types)
INSTALL(FILES ${INPUTS_DIST_FILES} DESTINATION input-types)
