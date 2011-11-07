#
# Input type plugins 
#
include_directories(AFTER ${PCAP_INCLUDE_DIRS})

set(NORMALIZE_FILTERS_DIST_FILES
)

macro(declare_library target)
	get_target_property(LIB_NAME ${target} LOCATION)
	set(NORMALIZE_FILTERS_DIST_LIST
		${NORMALIZE_FILTERS_DIST_LIST}
		${target})
	set(_NORMALIZE_FILTERS_LIB_LIST
		${_NORMALIZE_FILTERS_LIB_LIST}
		${LIB_NAME})
endmacro(declare_library)

# Splitters

#add_library(normalize_filter_splitter_regexp SHARED splitter_regexp/PVFieldSplitterRegexp.cpp splitter_regexp/PVFieldSplitterRegexp.h splitter_regexp/register.cpp)
#target_link_libraries(normalize_filter_splitter_regexp pvfilter pvcore)

SET(CSV_MOC_HEADERS
splitter_csv/PVFieldSplitterCSVParamWidget.h
splitter_csv/qkeysequencewidget/qkeysequencewidget.h
splitter_csv/qkeysequencewidget/qkeysequencewidget_p.h
)
QT4_WRAP_CPP(CSV_MOC_SRCS ${CSV_MOC_HEADERS})
add_library(normalize_filter_splitter_csv SHARED splitter_csv/PVFieldSplitterCSV.cpp splitter_csv/PVFieldSplitterCSV.h splitter_csv/PVFieldSplitterCSVParamWidget.cpp splitter_csv/qkeysequencewidget/qkeysequencewidget.cpp ${CSV_MOC_SRCS} splitter_csv/libcsv.cpp splitter_csv/register.cpp)
target_link_libraries(normalize_filter_splitter_csv ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})
declare_library(normalize_filter_splitter_csv)

SET(REGEXP_MOC_HEADERS
splitter_regexp/PVFieldSplitterRegexpParamWidget.h
)
QT4_WRAP_CPP(REGEXP_MOC_SRCS ${REGEXP_MOC_HEADERS})
add_library(normalize_filter_splitter_regexp SHARED splitter_regexp/PVFieldSplitterRegexp.cpp splitter_regexp/PVFieldSplitterRegexp.h splitter_regexp/PVFieldSplitterRegexpParamWidget.cpp ${REGEXP_MOC_SRCS} splitter_regexp/register.cpp)
target_link_libraries(normalize_filter_splitter_regexp ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES} ${Boost_LIBRARIES} ${ICU_LIBRARY})
declare_library(normalize_filter_splitter_regexp)

add_library(normalize_filter_splitter_url SHARED splitter_url/PVFieldSplitterURL.cpp splitter_url/PVFieldSplitterURL.h splitter_url/register.cpp)
target_link_libraries(normalize_filter_splitter_url ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})
declare_library(normalize_filter_splitter_url)

add_library(normalize_filter_splitter_pcap SHARED splitter_pcap/PVFieldSplitterPcapPacket.cpp splitter_pcap/PVFieldSplitterPcapPacket.h splitter_pcap/register.cpp)
target_link_libraries(normalize_filter_splitter_pcap ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES} ${PCAP_LIBRARIES} ${WINSOCK_LIB})
declare_library(normalize_filter_splitter_pcap)

# "grep"

add_library(normalize_filter_grep_regexp SHARED grep_regexp/PVFieldFilterRegexpGrep.cpp grep_regexp/PVFieldFilterRegexpGrep.h grep_regexp/register.cpp)
target_link_libraries(normalize_filter_grep_regexp ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})
declare_library(normalize_filter_grep_regexp)

INSTALL(TARGETS ${NORMALIZE_FILTERS_DIST_LIST} LIBRARY DESTINATION normalize-filters
                                               RUNTIME DESTINATION normalize-filters)
INSTALL(FILES ${NORMALIZE_FILTERS_DIST_FILES} DESTINATION normalize-filters)

SET(NORMALIZE_FILTERS_LIB_LIST ${_NORMALIZE_FILTERS_LIB_LIST} PARENT_SCOPE)
