SET(SRC_FILES
PVFilterLibrary.cpp
PVChunkFilter.cpp
PVChunkFilterByElt.cpp
PVChunkFilterSource.cpp
PVChunkFilterCountElts.cpp
PVElementFilterByFields.cpp
PVElementFilterGrep.cpp
PVElementFilterRandInvalid.cpp
PVFieldFilterGrep.cpp
PVFieldsFilter.cpp
PVFieldsMappingFilter.cpp
PVFieldSplitterChar.cpp
PVFieldSplitterUTF16Char.cpp
PVFieldsFilterParamWidget.cpp
PVPluginsLoad.cpp
PVRawSourceBase.cpp
)

SET(HEADER_FILES
include/pvfilter/PVChunkFilter.h
include/pvfilter/PVChunkFilterByElt.h
include/pvfilter/PVChunkFilterSource.h
include/pvfilter/PVChunkFilterCountElts.h
include/pvfilter/PVElementFilterByFields.h
include/pvfilter/PVElementFilter.h
include/pvfilter/PVFieldsFilter.h
include/pvfilter/PVFilterFunction.h
include/pvfilter/PVFilterLibrary.h
include/pvfilter/PVElementFilterGrep.h
include/pvfilter/PVElementFilterRandInvalid.h
include/pvfilter/PVFieldFilterGrep.h
include/pvfilter/PVFieldsFilterParamWidget.h
include/pvfilter/PVFieldSplitterChar.h
include/pvfilter/PVFieldSplitterUTF16Char.h
include/pvfilter/PVFieldsMappingFilter.h
include/pvfilter/PVPluginsLoad.h
include/pvfilter/PVRawSourceBase.h
)

add_library(pvfilter SHARED ${SRC_FILES} ${HEADER_FILES})
set_target_properties(pvfilter PROPERTIES SOVERSION 1)

target_link_libraries(pvfilter ${QT_LIBRARIES} ${PVCORE_LIBRARIES} ${TBB_LIBRARIES} ${TBB_MALLOC_LIBRARY} ${Boost_LIBRARIES}  ${ICU_LIBRARY})

INSTALL(TARGETS pvfilter DESTINATION .)

