#
# Input type plugins 
#
include_directories(AFTER ${PCAP_INCLUDE_DIRS})

set(NORMALIZE_FILTERS_DIST_LIST
normalize_filter_splitter_regexp
normalize_filter_splitter_csv
normalize_filter_splitter_url
normalize_filter_splitter_pcap
normalize_filter_grep_regexp
)

set(NORMALIZE_FILTERS_DIST_FILES
)

# Splitters

#add_library(normalize_filter_splitter_regexp SHARED splitter_regexp/PVFieldSplitterRegexp.cpp splitter_regexp/PVFieldSplitterRegexp.h splitter_regexp/register.cpp)
#target_link_libraries(normalize_filter_splitter_regexp pvfilter pvcore)

SET(CSV_MOC_HEADERS
splitter_csv/PVFieldSplitterCSVParamWidget.h
)
QT4_WRAP_CPP(CSV_MOC_SRCS ${CSV_MOC_HEADERS})
add_library(normalize_filter_splitter_csv SHARED splitter_csv/PVFieldSplitterCSV.cpp splitter_csv/PVFieldSplitterCSV.h splitter_csv/PVFieldSplitterCSVParamWidget.cpp ${CSV_MOC_SRCS} splitter_csv/libcsv.cpp splitter_csv/register.cpp)
target_link_libraries(normalize_filter_splitter_csv ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})

SET(REGEXP_MOC_HEADERS
splitter_regexp/PVFieldSplitterRegexpParamWidget.h
)
QT4_WRAP_CPP(REGEXP_MOC_SRCS ${REGEXP_MOC_HEADERS})
add_library(normalize_filter_splitter_regexp SHARED splitter_regexp/PVFieldSplitterRegexp.cpp splitter_regexp/PVFieldSplitterRegexp.h splitter_regexp/PVFieldSplitterRegexpParamWidget.cpp ${REGEXP_MOC_SRCS} splitter_regexp/register.cpp)
target_link_libraries(normalize_filter_splitter_regexp ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})


add_library(normalize_filter_splitter_url SHARED splitter_url/PVFieldSplitterURL.cpp splitter_url/PVFieldSplitterURL.h splitter_url/register.cpp)
target_link_libraries(normalize_filter_splitter_url ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})

add_library(normalize_filter_splitter_pcap SHARED splitter_pcap/PVFieldSplitterPcapPacket.cpp splitter_pcap/PVFieldSplitterPcapPacket.h splitter_pcap/register.cpp)
target_link_libraries(normalize_filter_splitter_pcap ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES} ${PCAP_LIBRARIES} ${DUMBNET_LIBRARIES})

# "grep"

add_library(normalize_filter_grep_regexp SHARED grep_regexp/PVFieldFilterRegexpGrep.cpp grep_regexp/PVFieldFilterRegexpGrep.h grep_regexp/register.cpp)
target_link_libraries(normalize_filter_grep_regexp ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})


INSTALL(TARGETS ${NORMALIZE_FILTERS_DIST_LIST} DESTINATION normalize-filters)
INSTALL(FILES ${NORMALIZE_FILTERS_DIST_FILES} DESTINATION normalize-filters)
