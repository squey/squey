#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

configure_file(test_issues.sh.cmake test_issues.sh @ONLY)

set(LINK_LIBRARIES pvkernel ${TBB_LIBRARIES} ${ICU_LIBRARY})

# Temporary output directory for NRAW disk backend based tests
if (NOT DEFINED PVKERNEL_RUSH_TEST_NRAW_TMP)
	message(STATUS "PVKERNEL_RUSH_TEST_NRAW_TMP not defined, /tmp will be used by default")
	set(PVKERNEL_RUSH_TEST_NRAW_TMP "/tmp")
endif()

set(PVRUSH_TEST_FILES "${TESTS_FILES_DIR}/pvkernel/rush")
set(PVRUSH_TEST_FILES_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/test-files")

# Test UTF8 conversion
add_pv_compile_test(Trush_conv_utf8 LINKS ${LINK_LIBRARIES} FILES conv_utf8.cpp helpers.cpp)
add_bench(Trush_conv_utf8 LINKS ${LINK_LIBRARIES} FILES conv_utf8.cpp helpers.cpp)

macro(PV_ADD_TEST_CONV_UTF8 input_file chunks_size)
	add_test(TestConvUtf8_${input_file}_${chunks_size} Trush_conv_utf8 ${PVRUSH_TEST_FILES}/${input_file} ${PVRUSH_TEST_FILES}/${input_file}.out ${chunks_size})
endmacro()

PV_ADD_TEST_CONV_UTF8("charset/utf8" 20000)
PV_ADD_TEST_CONV_UTF8("charset/utf16" 20000)
PV_ADD_TEST_CONV_UTF8("charset/utf32" 20000)
PV_ADD_TEST_CONV_UTF8("charset/latin1" 20000)
PV_ADD_TEST_CONV_UTF8("conv_utf8_align" 20000)
# TODO : Re-enable this test once we support chunk growing when they are too small.
#PV_ADD_TEST_CONV_UTF8("conv_utf8_align" 200)

# Test the aggregator
add_pv_compile_test(Trush_agg LINKS ${LINK_LIBRARIES} FILES agg.cpp helpers.cpp)
add_test(TestAggregator1 Trush_agg 6000 ${PVRUSH_TEST_FILES}/aggregator/set1)
add_test(TestAggregator2 Trush_agg 6000 ${PVRUSH_TEST_FILES}/aggregator/set2)

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

# Test for filters
add_pv_compile_test(Trush_filter_grep LINKS ${LINK_LIBRARIES} FILES filter_grep.cpp helpers.cpp)
add_test(Trush_filter_grep Trush_filter_grep)
add_bench(Trush_filter_grep LINKS ${LINK_LIBRARIES} FILES filter_grep.cpp helpers.cpp)

# Test for converter
add_pv_compile_test(Trush_converter_GUID_to_IPV4 LINKS ${LINK_LIBRARIES} FILES converter_GUID_to_IPV4.cpp helpers.cpp)
add_test(Trush_converter_GUID_to_IPV4 Trush_converter_GUID_to_IPV4)
add_bench(Trush_converter_GUID_to_IPV4 LINKS ${LINK_LIBRARIES} FILES converter_GUID_to_IPV4.cpp helpers.cpp)

add_pv_compile_test(Trush_converter_GUID_to_IPV6 LINKS ${LINK_LIBRARIES} FILES converter_GUID_to_IPV6.cpp helpers.cpp)
add_test(Trush_converter_GUID_to_IPV6 Trush_converter_GUID_to_IPV6)
add_bench(Trush_converter_GUID_to_IPV6 LINKS ${LINK_LIBRARIES} FILES converter_GUID_to_IPV6.cpp helpers.cpp)

add_pv_compile_test(Trush_converter_substitution LINKS ${LINK_LIBRARIES} FILES converter_substitution.cpp helpers.cpp)
add_test(Trush_converter_substitution Trush_converter_substitution)
add_bench(Trush_converter_substitution LINKS ${LINK_LIBRARIES} FILES converter_substitution.cpp helpers.cpp)

# Test for splitters
add_pv_compile_test(Trush_splitter_key_value LINKS ${LINK_LIBRARIES} FILES splitter_key_value.cpp helpers.cpp)
add_test(Trush_splitter_key_value Trush_splitter_key_value)
add_bench(Trush_splitter_key_value LINKS ${LINK_LIBRARIES} FILES splitter_key_value.cpp helpers.cpp)

add_pv_compile_test(Trush_splitter_dns_fqdn LINKS ${LINK_LIBRARIES} FILES splitter_dns_fqdn.cpp helpers.cpp)
add_test(Trush_splitter_dns_fqdn Trush_splitter_dns_fqdn)
add_bench(Trush_splitter_dns_fqdn LINKS ${LINK_LIBRARIES} FILES splitter_dns_fqdn.cpp helpers.cpp)

add_pv_compile_test(Trush_splitter_mac LINKS ${LINK_LIBRARIES} FILES splitter_mac.cpp helpers.cpp)
add_test(Trush_splitter_mac Trush_splitter_mac)
add_bench(Trush_splitter_mac LINKS ${LINK_LIBRARIES} FILES splitter_mac.cpp helpers.cpp)

add_pv_compile_test(Trush_splitter_ipv4 LINKS ${LINK_LIBRARIES} FILES splitter_ipv4.cpp helpers.cpp)
add_test(Trush_splitter_ipv4 Trush_splitter_ipv4)
add_bench(Trush_splitter_ipv4 LINKS ${LINK_LIBRARIES} FILES splitter_ipv4.cpp helpers.cpp)

add_pv_compile_test(Trush_splitter_ipv6 LINKS ${LINK_LIBRARIES} FILES splitter_ipv6.cpp helpers.cpp)
add_test(Trush_splitter_ipv6 Trush_splitter_ipv6)
add_bench(Trush_splitter_ipv6 LINKS ${LINK_LIBRARIES} FILES splitter_ipv6.cpp helpers.cpp)

add_pv_compile_test(Trush_splitter_regexp LINKS ${LINK_LIBRARIES} FILES splitter_regexp.cpp helpers.cpp)
add_test(Trush_splitter_regexp Trush_splitter_regexp)
add_bench(Trush_splitter_regexp LINKS ${LINK_LIBRARIES} FILES splitter_regexp.cpp helpers.cpp)

add_pv_compile_test(Trush_splitter_regexp_full_line LINKS ${LINK_LIBRARIES} FILES splitter_regexp_full_line.cpp helpers.cpp)
add_test(Trush_splitter_regexp_full_line Trush_splitter_regexp_full_line)
add_bench(Trush_splitter_regexp_full_line LINKS ${LINK_LIBRARIES} FILES splitter_regexp_full_line.cpp helpers.cpp)

add_pv_compile_test(Trush_splitter_csv LINKS ${LINK_LIBRARIES} FILES splitter_csv.cpp helpers.cpp)
add_test(Trush_splitter_csv Trush_splitter_csv)
add_bench(Trush_splitter_csv LINKS ${LINK_LIBRARIES} FILES splitter_csv.cpp helpers.cpp)

add_pv_compile_test(Trush_splitter_url LINKS ${LINK_LIBRARIES} FILES splitter_url.cpp helpers.cpp)
add_test(Trush_splitter_url Trush_splitter_url)
add_bench(Trush_splitter_url LINKS ${LINK_LIBRARIES} FILES splitter_url.cpp helpers.cpp)

add_pv_compile_test(Trush_splitter_duplicate LINKS ${LINK_LIBRARIES} FILES splitter_duplicate.cpp helpers.cpp)
add_test(Trush_splitter_duplicate Trush_splitter_duplicate)
add_bench(Trush_splitter_duplicate LINKS ${LINK_LIBRARIES} FILES splitter_duplicate.cpp helpers.cpp)

add_pv_compile_test(Trush_filter_composition LINKS ${LINK_LIBRARIES} FILES filter_composition.cpp helpers.cpp)
add_test(Trush_filter_composition Trush_filter_composition)
add_bench(Trush_filter_composition LINKS ${LINK_LIBRARIES} FILES filter_composition.cpp helpers.cpp)

# Plugin sources
add_pv_compile_test(Trush_splunk_source LINKS ${LINK_LIBRARIES}
	FILES splunk_source.cpp helpers.cpp
	../../plugins/common/splunk/PVSplunkInfos.cpp
	../../plugins/common/splunk/PVSplunkQuery.cpp)
add_test(Trush_splunk_source Trush_splunk_source)
add_bench(Trush_splunk_source LINKS ${LINK_LIBRARIES}
	FILES splunk_source.cpp helpers.cpp
	../../plugins/common/splunk/PVSplunkInfos.cpp
	../../plugins/common/splunk/PVSplunkQuery.cpp)

add_pv_compile_test(Trush_elasticsearch_source LINKS ${LINK_LIBRARIES}
	FILES elasticsearch_source.cpp helpers.cpp
	../../plugins/common/elasticsearch/PVElasticsearchInfos.cpp
	../../plugins/common/elasticsearch/PVElasticsearchQuery.cpp)
add_test(Trush_elasticsearch_source Trush_elasticsearch_source)
add_bench(Trush_elasticsearch_source LINKS ${LINK_LIBRARIES}
	FILES elasticsearch_source.cpp helpers.cpp
	../../plugins/common/elasticsearch/PVElasticsearchInfos.cpp
	../../plugins/common/elasticsearch/PVElasticsearchQuery.cpp)

# Plugin input tests
add_pv_compile_test(Trush_mem LINKS ${LINK_LIBRARIES} FILES bench_memory.cpp)
add_pv_compile_test(Trush_filter_guess LINKS ${LINK_LIBRARIES} FILES filter_guess.cpp)
add_pv_compile_test(Trush_parser LINKS ${LINK_LIBRARIES} FILES parser.cpp)
add_pv_compile_test(Trush_format_conv LINKS ${LINK_LIBRARIES} FILES format_conv.cpp)

# Check NRaw fill and dump
add_pv_compile_test(Trush_nraw_create LINKS ${LINK_LIBRARIES} FILES nraw_create.cpp)
add_test(Trush_nraw_create Trush_nraw_create)
add_bench(Trush_nraw_create LINKS ${LINK_LIBRARIES} FILES nraw_create.cpp)

add_pv_compile_test(Trush_nraw_dump LINKS ${LINK_LIBRARIES} FILES nraw_dump.cpp)
add_test(Trush_nraw_dump Trush_nraw_dump)
add_bench(Trush_nraw_dump LINKS ${LINK_LIBRARIES} FILES nraw_dump.cpp)

add_pv_compile_test(Trush_format_dump LINKS ${LINK_LIBRARIES} FILES format_dump.cpp helpers.cpp)
add_pv_compile_test(Trush_process_file_auto_discovery LINKS ${LINK_LIBRARIES} FILES process_file_auto_discovery.cpp helpers.cpp)

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
