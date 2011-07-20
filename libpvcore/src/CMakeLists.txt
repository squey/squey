include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpvcore/src/include)
include_directories(AFTER ${ICU_INCLUDE})

set(SRC_FILES
debug.cpp
environ.cpp
network.cpp
sysinfo.cpp

PVArgument.cpp
PVAxisIndexType.cpp
PVBufferSlice.cpp
PVClassLibrary.cpp
PVDateTimeParser.cpp
PVElement.cpp
PVField.cpp
PVListFloat2D.cpp
PVLogger.cpp
PVPCRESplitter.cpp
PVSplitter.cpp
PVXmlParamParser.cpp
PVXmlTreeNodeDom.cpp
)

set(HEADER_FILES
include/pvcore/debug.h
include/pvcore/environ.h
include/pvcore/export.h
include/pvcore/general.h
include/pvcore/network.h
include/pvcore/pv_pcre_splitter.h
include/pvcore/pv_splitter.h
include/pvcore/string.h
include/pvcore/sysinfo.h
include/pvcore/types.h

include/pvcore/PVArgument.h
include/pvcore/PVAxisIndexType.h
include/pvcore/PVAxesIndexType.h
include/pvcore/PVClassLibrary.h
include/pvcore/PVDateTimeParser.h
include/pvcore/PVEnumType.h
include/pvcore/PVColorGradientDualSliderType.h
include/pvcore/PVBufferSlice.h
include/pvcore/PVChunk.h
include/pvcore/PVElement.h
include/pvcore/PVElementData.h
include/pvcore/PVField.h
include/pvcore/PVListFloat2D.h
include/pvcore/PVLogger.h
include/pvcore/PVMeanValue.h
include/pvcore/PVRegistrableClass.h
include/pvcore/PVSpinBoxType.h
include/pvcore/PVTBBMaxArray.h
include/pvcore/PVTBBMinMaxArray.h
include/pvcore/PVXmlParamParser.h
include/pvcore/PVXmlParamParserData.h
include/pvcore/PVXmlTreeNodeDom.h 
)

SET(MOC_HEADERS
)
QT4_WRAP_CPP(MOC_SRC ${MOC_HEADERS})

add_library(pvcore SHARED ${SRC_FILES} ${HEADER_FILES} ${MOC_SRC} ${MOC_HEADERS})

set_target_properties(pvcore PROPERTIES SOVERSION 1)

target_link_libraries(pvcore ${QT_LIBRARIES} ${DUMBNET_LIBRARIES} ${ICU_LIBRARY} ${TBB_LIBRARIES} ${TBB_MALLOC_LIBRARY})

INSTALL(TARGETS pvcore DESTINATION .)
