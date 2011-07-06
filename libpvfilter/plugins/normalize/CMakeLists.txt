#
# Input type plugins 
#
include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpvcore/src/include)
include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpvfilter/src/include)
include_directories(AFTER ${PCAP_INCLUDE_DIRS})

link_directories(${Picviz_Inspector_SOURCE_DIR}/libpvfilter/src/ ${Picviz_Inspector_SOURCE_DIR}/libpvcore/src/)

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

add_library(normalize_filter_splitter_regexp SHARED splitter_regexp/PVFieldSplitterRegexp.cpp splitter_regexp/PVFieldSplitterRegexp.h splitter_regexp/register.cpp)
target_link_libraries(normalize_filter_splitter_regexp pvfilter pvcore)

add_library(normalize_filter_splitter_csv SHARED splitter_csv/PVFieldSplitterCSV.cpp splitter_csv/PVFieldSplitterCSV.h splitter_csv/libcsv.cpp splitter_csv/register.cpp)
target_link_libraries(normalize_filter_splitter_csv pvfilter pvcore)

add_library(normalize_filter_splitter_url SHARED splitter_url/PVFieldSplitterURL.cpp splitter_url/PVFieldSplitterURL.h splitter_url/register.cpp)
target_link_libraries(normalize_filter_splitter_url pvfilter pvcore)

add_library(normalize_filter_splitter_pcap SHARED splitter_pcap/PVFieldSplitterPcapPacket.cpp splitter_pcap/PVFieldSplitterPcapPacket.h splitter_pcap/register.cpp)
target_link_libraries(normalize_filter_splitter_pcap pvfilter pvcore ${PCAP_LIBRARIES} ${DUMBNET_LIBRARIES})

# "grep"

add_library(normalize_filter_grep_regexp SHARED grep_regexp/PVFieldFilterRegexpGrep.cpp grep_regexp/PVFieldFilterRegexpGrep.h grep_regexp/register.cpp)
target_link_libraries(normalize_filter_grep_regexp pvfilter pvcore)


INSTALL(TARGETS ${NORMALIZE_FILTERS_DIST_LIST} DESTINATION normalize-filters)
INSTALL(FILES ${NORMALIZE_FILTERS_DIST_FILES} DESTINATION normalize-filters)
