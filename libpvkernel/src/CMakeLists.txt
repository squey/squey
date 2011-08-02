include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpvkernel/src/include)
include_directories(AFTER ${ICU_INCLUDE})

set(CORE_SRC_FILES
core/debug.cpp
core/environ.cpp
core/network.cpp
core/sysinfo.cpp

core/PVArgument.cpp
core/PVAxisIndexType.cpp
core/PVAxisIndexCheckBoxType.cpp
core/PVBufferSlice.cpp
core/PVClassLibrary.cpp
core/PVDateTimeParser.cpp
core/PVElement.cpp
core/PVField.cpp
core/PVListFloat2D.cpp
core/PVLogger.cpp
core/PVPCRESplitter.cpp
core/PVSplitter.cpp
)

set(CORE_HEADER_FILES
include/pvkernel/core/debug.h
include/pvkernel/core/environ.h
include/pvkernel/core/export.h
include/pvkernel/core/general.h
include/pvkernel/core/network.h
include/pvkernel/core/pv_pcre_splitter.h
include/pvkernel/core/pv_splitter.h
include/pvkernel/core/string.h
include/pvkernel/core/sysinfo.h
include/pvkernel/core/types.h

include/pvkernel/core/PVArgument.h
include/pvkernel/core/PVAxisIndexType.h
include/pvkernel/core/PVAxesIndexType.h
include/pvkernel/core/PVClassLibrary.h
include/pvkernel/core/PVDateTimeParser.h
include/pvkernel/core/PVEnumType.h
include/pvkernel/core/PVColorGradientDualSliderType.h
include/pvkernel/core/PVBufferSlice.h
include/pvkernel/core/PVChunk.h
include/pvkernel/core/PVElement.h
include/pvkernel/core/PVElementData.h
include/pvkernel/core/PVField.h
include/pvkernel/core/PVListFloat2D.h
include/pvkernel/core/PVLogger.h
include/pvkernel/core/PVMeanValue.h
include/pvkernel/core/PVRegistrableClass.h
include/pvkernel/core/PVSpinBoxType.h
include/pvkernel/core/PVTBBMaxArray.h
include/pvkernel/core/PVTBBMinMaxArray.h
)

SET(MOC_HEADERS
)
QT4_WRAP_CPP(MOC_SRC ${MOC_HEADERS})

add_library(pvkernel SHARED ${CORE_SRC_FILES} ${CORE_HEADER_FILES} ${MOC_SRC} ${MOC_HEADERS})

set_target_properties(pvkernel PROPERTIES SOVERSION 1)

target_link_libraries(pvkernel ${QT_LIBRARIES} ${DUMBNET_LIBRARIES} ${ICU_LIBRARY} ${TBB_LIBRARIES} ${TBB_MALLOC_LIBRARY} ${OpenMP_LIBRARIES})

INSTALL(TARGETS pvkernel DESTINATION .)
