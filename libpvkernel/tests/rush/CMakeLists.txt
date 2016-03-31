#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

configure_file(test_issues.sh.cmake test_issues.sh @ONLY)
configure_file(test_splitter_duplicate.sh.cmake test_splitter_duplicate.sh @ONLY)
configure_file(test_splitter_pcap.sh.cmake test_splitter_pcap.sh @ONLY)
configure_file(test_splitter_regexp.sh.cmake test_splitter_regexp.sh @ONLY)
configure_file(test_splitter_url.sh.cmake test_splitter_url.sh @ONLY)
configure_file(test_grep_regexp.sh.cmake test_grep_regexp.sh @ONLY)

set(LINK_LIBRARIES pvkernel ${TBB_LIBRARIES} ${ICU_LIBRARY})

# Temporary output directory for NRAW disk backend based tests
if (NOT DEFINED PVKERNEL_RUSH_TEST_NRAW_TMP)
	message(STATUS "PVKERNEL_RUSH_TEST_NRAW_TMP not defined, /tmp will be used by default")
	set(PVKERNEL_RUSH_TEST_NRAW_TMP "/tmp")
endif()

set(PVRUSH_TEST_FILES "${TESTS_FILES_DIR}/pvkernel/rush")
set(PVRUSH_TEST_FILES_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/test-files")

# Test UTF16 conversion
add_pv_compile_test(Trush_conv_utf16 LINKS ${LINK_LIBRARIES} FILES conv_utf16.cpp helpers.cpp)

macro(PV_ADD_TEST_CONV_UTF16 charset)
	add_test(TestConvUtf16_${charset} ${TEST_DIFF_STDOUT}
		${PVRUSH_TEST_FILES}/charset/${charset}.out
		${PVRUSH_TEST_FILES_OUTPUT}/charset/${charset}.diff
		${CMAKE_CURRENT_BINARY_DIR}/Trush_conv_utf16 ${PVRUSH_TEST_FILES}/charset/${charset} 20000)
endmacro()

PV_ADD_TEST_CONV_UTF16("utf8")
PV_ADD_TEST_CONV_UTF16("utf16")
PV_ADD_TEST_CONV_UTF16("utf32")
PV_ADD_TEST_CONV_UTF16("latin1")

# Test UTF16 conversion and alignement on a newline (PVUnicodeSource object)
add_pv_compile_test(Trush_conv_utf16_align LINKS ${LINK_LIBRARIES} FILES conv_utf16_align.cpp helpers.cpp)

macro(PV_ADD_TEST_CONV_UTF16_ALIGN CASE CHUNK_SIZE)
	add_test(TestConvUtf16Align_${CASE}_${CHUNK_SIZE} ${TEST_DIFF_STDOUT}
		${PVRUSH_TEST_FILES}/${CASE}.out
		${PVRUSH_TEST_FILES_OUTPUT}/${CASE}.diff
		${CMAKE_CURRENT_BINARY_DIR}/Trush_conv_utf16_align
		${PVRUSH_TEST_FILES}/${CASE} ${CHUNK_SIZE})
endmacro()

PV_ADD_TEST_CONV_UTF16_ALIGN(conv_utf16_align 20000)
PV_ADD_TEST_CONV_UTF16_ALIGN(conv_utf16_align 2000)
# FIXME : Align files do not exist
#PV_ADD_TEST_CONV_UTF16_ALIGN(conv_utf16_align_newline 20000)
#PV_ADD_TEST_CONV_UTF16_ALIGN(conv_utf16_align_newline 2000)

# Test the aggregator
add_pv_compile_test(Trush_agg LINKS ${LINK_LIBRARIES} FILES agg.cpp helpers.cpp)
add_test(TestAggregator1 ${TEST_DIFF_STDOUT}
	${PVRUSH_TEST_FILES}/aggregator/set1.out
	${PVRUSH_TEST_FILES_OUTPUT}/aggregator/set1.diff
	${CMAKE_CURRENT_BINARY_DIR}/Trush_agg 6000
	${PVRUSH_TEST_FILES}/aggregator/set1)
add_test(TestAggregator2 ${TEST_DIFF_STDOUT}
	${PVRUSH_TEST_FILES}/aggregator/set2.out
	${PVRUSH_TEST_FILES_OUTPUT}/aggregator/set2.diff
	${CMAKE_CURRENT_BINARY_DIR}/Trush_agg 6000
	${PVRUSH_TEST_FILES}/aggregator/set2)

# Test the aggregator in strict mode
add_pv_compile_test(Trush_agg_strict LINKS ${LINK_LIBRARIES} FILES agg_strict.cpp helpers.cpp)
add_test(TestAggregatorStrict1 ${TEST_DIFF_STDOUT}
	${PVRUSH_TEST_FILES}/aggregator/set1.strict.out
	${PVRUSH_TEST_FILES_OUTPUT}/aggregator/set1.strict.diff
	${CMAKE_CURRENT_BINARY_DIR}/Trush_agg_strict 6000
	${PVRUSH_TEST_FILES}/aggregator/set1)
add_test(TestAggregatorStrict2 ${TEST_DIFF_STDOUT}
	${PVRUSH_TEST_FILES}/aggregator/set2.strict.out
	${PVRUSH_TEST_FILES_OUTPUT}/aggregator/set2.strict.diff
	${CMAKE_CURRENT_BINARY_DIR}/Trush_agg_strict 6000
	${PVRUSH_TEST_FILES}/aggregator/set2)

# Test the extractor
add_pv_compile_test(Trush_splitter_regexp LINKS ${LINK_LIBRARIES} FILES splitter_regexp.cpp helpers.cpp)
add_test(Trush_splitter_regexp ${CMAKE_CURRENT_BINARY_DIR}/test_splitter_regexp.sh)
add_pv_compile_test(Trush_splitter_csv LINKS ${LINK_LIBRARIES} FILES splitter_csv.cpp helpers.cpp)
add_test(Trush_splitter_csv Trush_splitter_csv)
add_bench(Trush_splitter_csv LINKS ${LINK_LIBRARIES} FILES splitter_csv.cpp helpers.cpp)

add_pv_compile_test(Trush_splitter_url LINKS ${LINK_LIBRARIES} FILES splitter_url.cpp helpers.cpp)
add_test(Trush_splitter_url ${CMAKE_CURRENT_BINARY_DIR}/test_splitter_url.sh)
add_pv_compile_test(Trush_splitter_pcap LINKS ${LINK_LIBRARIES} FILES splitter_pcap.cpp helpers.cpp)
add_test(Trush_splitter_pcap ${CMAKE_CURRENT_BINARY_DIR}/test_splitter_pcap.sh)
add_pv_compile_test(Trush_splitter_regexp LINKS ${LINK_LIBRARIES} FILES splitter_regexp.cpp helpers.cpp)
add_test(Trush_splitter_regexp ${CMAKE_CURRENT_BINARY_DIR}/test_splitter_regexp.sh)
add_pv_compile_test(Trush_splitter_duplicate LINKS ${LINK_LIBRARIES} FILES splitter_duplicate.cpp helpers.cpp)
add_test(Trush_splitter_duplicate ${CMAKE_CURRENT_BINARY_DIR}/test_splitter_duplicate.sh)
add_pv_compile_test(Trush_filter_composition LINKS ${LINK_LIBRARIES} FILES filter_composition.cpp helpers.cpp)
add_test(TestFilterComposition ${TEST_DIFF_STDOUT}
	${PVRUSH_TEST_FILES}/filter_composition.out
	${PVRUSH_TEST_FILES}/filter_composition.diff
	${CMAKE_CURRENT_BINARY_DIR}/Trush_filter_composition
	${PVRUSH_TEST_FILES}/filter_composition 6000)

# Plugin input tests
add_pv_compile_test(Trush_mem LINKS ${LINK_LIBRARIES} FILES bench_memory.cpp)
add_pv_compile_test(Trush_filter_guess LINKS ${LINK_LIBRARIES} FILES filter_guess.cpp)
add_pv_compile_test(Trush_parser LINKS ${LINK_LIBRARIES} FILES parser.cpp)
add_pv_compile_test(Trush_format_conv LINKS ${LINK_LIBRARIES} FILES format_conv.cpp)
add_pv_compile_test(Trush_nraw_create LINKS ${LINK_LIBRARIES} FILES nraw_create.cpp)
add_test(Trush_nraw_create Trush_nraw_create)
add_bench(Trush_nraw_create LINKS ${LINK_LIBRARIES} FILES nraw_create.cpp)
add_pv_compile_test(Trush_nraw_dump LINKS ${LINK_LIBRARIES} FILES nraw_dump.cpp)
add_test(Trush_nraw_dump Trush_nraw_dump)
add_bench(Trush_nraw_dump LINKS ${LINK_LIBRARIES} FILES nraw_dump.cpp)

add_pv_compile_test(Trush_format_dump LINKS ${LINK_LIBRARIES} FILES format_dump.cpp helpers.cpp)
add_pv_compile_test(Trush_process_file_auto_discovery LINKS ${LINK_LIBRARIES} FILES process_file_auto_discovery.cpp helpers.cpp)
add_pv_compile_test(arcsight_src LINKS ${LINK_LIBRARIES} FILES test_arcsight_source.cpp helpers.cpp
	${CMAKE_SOURCE_DIR}/libpvkernel/plugins/common/arcsight/PVArcsightInfos.cpp
	${CMAKE_SOURCE_DIR}/libpvkernel/plugins/common/arcsight/PVArcsightQuery.cpp)

add_pv_compile_test(Trush_format LINKS ${LINK_LIBRARIES} FILES format.cpp helpers.cpp)
add_test(Trush_format Trush_format ${PVRUSH_TEST_FILES}/formats)

add_pv_compile_test(Trush_ticket_1 LINKS ${LINK_LIBRARIES} FILES ticket_1.cpp)
add_test(Trush_ticket_1 Trush_ticket_1 ${PVRUSH_TEST_FILES})

add_pv_compile_test(Trush_ticket_2 LINKS ${LINK_LIBRARIES} FILES ticket_2.cpp)
add_test(Trush_ticket_2 Trush_ticket_2)

add_pv_compile_test(Trush_ticket_28 LINKS ${LINK_LIBRARIES} FILES ticket_28.cpp)
add_test(Trush_ticket_28 Trush_ticket_28 ${PVRUSH_TEST_FILES})

add_pv_compile_test(Trush_nraw_unused_directories LINKS ${LINK_LIBRARIES} FILES nraw_unused_directories.cpp)
add_test(Trush_nraw_unused_directories Trush_nraw_unused_directories ${PVKERNEL_RUSH_TEST_NRAW_TMP})

# Elasticsearch plugin test
set(ELASTICSEARCH_SRC_FILES
../../plugins/common/elasticsearch/PVElasticsearchAPI.cpp
../../plugins/common/elasticsearch/PVElasticsearchInfos.cpp
../../plugins/common/elasticsearch/PVElasticsearchQuery.cpp
../../plugins/common/elasticsearch/PVElasticsearchJsonConverter.cpp
)
add_pv_compile_test(Trush_elasticsearch LINKS ${LINK_LIBRARIES} pvkernel curl FILES elasticsearch.cpp ${ELASTICSEARCH_SRC_FILES})
add_test(Trush_elasticsearch Trush_elasticsearch ${TESTS_FILES_DIR}/exports/Trush_elasticsearch)

# Splunk plugin test
set(SPLUNK_SRC_FILES
../../plugins/common/splunk/PVSplunkAPI.cpp
../../plugins/common/splunk/PVSplunkInfos.cpp
../../plugins/common/splunk/PVSplunkQuery.cpp
../../plugins/common/splunk/PVSplunkJsonConverter.cpp
)
add_pv_compile_test(Trush_splunk LINKS ${LINK_LIBRARIES} pvkernel curl FILES splunk.cpp ${SPLUNK_SRC_FILES})
add_test(Trush_splunk Trush_splunk ${TESTS_FILES_DIR}/exports/Trush_splunk)

# Datetime ICU to boost
add_pv_compile_test(Trush_datetime_support LINKS ${LINK_LIBRARIES} FILES datetime_support.cpp)
add_test(Trush_datetime_support Trush_datetime_support)
