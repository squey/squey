#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

set(LINK_LIBRARIES pvkernel ${TBB_LIBRARIES} ${ICU_LIBRARY})
add_definitions(-fopenmp)

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
# Check import can be done with too small chunk size.
PV_ADD_TEST_CONV_UTF8("conv_utf8_align" 200)
PV_ADD_TEST_CONV_UTF8("charset/windows1250" 10048576)
PV_ADD_TEST_CONV_UTF8("charset/utf8_windows1252" 10048576)

# Test the aggregator
add_pv_compile_test(Trush_agg LINKS ${LINK_LIBRARIES} FILES agg.cpp helpers.cpp)
add_test(TestAggregator1 Trush_agg 6000 ${PVRUSH_TEST_FILES}/aggregator/set1)
add_test(TestAggregator2 Trush_agg 6000 ${PVRUSH_TEST_FILES}/aggregator/set2)

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

add_pv_compile_test(Trush_converter_struct LINKS ${LINK_LIBRARIES} FILES converter_struct.cpp helpers.cpp)
add_test(Trush_converter_struct Trush_converter_struct)
add_bench(Trush_converter_struct LINKS ${LINK_LIBRARIES} FILES converter_struct.cpp helpers.cpp)

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

add_test(Trush_splitter_csv2 Trush_splitter_csv ${PVRUSH_TEST_FILES}/splitters/csv/file.csv ${PVRUSH_TEST_FILES}/splitters/csv/file.csv.out)

add_pv_compile_test(Trush_splitter_csv_nested LINKS ${LINK_LIBRARIES} FILES splitter_csv_nested.cpp helpers.cpp)
add_test(Trush_splitter_csv_nested Trush_splitter_csv_nested)

add_pv_compile_test(Trush_splitter_url LINKS ${LINK_LIBRARIES} FILES splitter_url.cpp helpers.cpp)
add_test(Trush_splitter_url Trush_splitter_url)
add_bench(Trush_splitter_url LINKS ${LINK_LIBRARIES} FILES splitter_url.cpp helpers.cpp)

add_pv_compile_test(Trush_splitter_duplicate LINKS ${LINK_LIBRARIES} FILES splitter_duplicate.cpp helpers.cpp)
add_test(Trush_splitter_duplicate Trush_splitter_duplicate)
add_bench(Trush_splitter_duplicate LINKS ${LINK_LIBRARIES} FILES splitter_duplicate.cpp helpers.cpp)

add_pv_compile_test(Trush_splitter_length LINKS ${LINK_LIBRARIES} FILES splitter_length.cpp helpers.cpp)
add_test(Trush_splitter_length Trush_splitter_length)

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
add_pv_compile_test(Trush_parser LINKS ${LINK_LIBRARIES} FILES parser.cpp)
add_pv_compile_test(Trush_format_conv LINKS ${LINK_LIBRARIES} FILES format_conv.cpp)

add_pv_compile_test(Trush_nraw_filtered LINKS ${LINK_LIBRARIES} FILES nraw_filtered.cpp)
add_test(Trush_nraw_filtered Trush_nraw_filtered)
add_bench(Trush_nraw_filtered LINKS ${LINK_LIBRARIES} FILES nraw_filtered.cpp)

# Check NRaw fill and dump
add_pv_compile_test(Trush_bad_splitter LINKS ${LINK_LIBRARIES} FILES bad_splitter.cpp)
add_test(Trush_bad_splitter Trush_bad_splitter)
add_bench(Trush_bad_splitter LINKS ${LINK_LIBRARIES} FILES bad_splitter.cpp)

add_pv_compile_test(Trush_nraw_create LINKS ${LINK_LIBRARIES} FILES nraw_create.cpp)
add_test(Trush_nraw_create Trush_nraw_create)
add_bench(Trush_nraw_create LINKS ${LINK_LIBRARIES} FILES nraw_create.cpp)

add_pv_compile_test(Trush_nraw_dump LINKS ${LINK_LIBRARIES} FILES nraw_dump.cpp)
add_test(Trush_nraw_dump Trush_nraw_dump)
add_bench(Trush_nraw_dump LINKS ${LINK_LIBRARIES} FILES nraw_dump.cpp)

add_pv_compile_test(Trush_process_file_auto_discovery LINKS ${LINK_LIBRARIES} FILES process_file_auto_discovery.cpp helpers.cpp)

add_pv_compile_test(Trush_format LINKS ${LINK_LIBRARIES} FILES format.cpp helpers.cpp)
add_test(Trush_format Trush_format ${PVRUSH_TEST_FILES}/formats)

add_pv_compile_test(Trush_ticket_1 LINKS ${LINK_LIBRARIES} FILES ticket_1.cpp)
add_test(Trush_ticket_1 Trush_ticket_1 ${PVRUSH_TEST_FILES})

add_pv_compile_test(Trush_ticket_2 LINKS ${LINK_LIBRARIES} FILES ticket_2.cpp)
add_test(LONG_Trush_ticket_2 Trush_ticket_2)

add_pv_compile_test(Trush_ticket_28 LINKS ${LINK_LIBRARIES} FILES ticket_28.cpp)
add_test(Trush_ticket_28 Trush_ticket_28 ${PVRUSH_TEST_FILES})

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

# Format ignore axes
add_pv_compile_test(Trush_format_import_export LINKS ${LINK_LIBRARIES} FILES format_import_export.cpp)
add_test(Trush_format_ignore_axes_csv Trush_format_import_export ${TESTS_FILES_DIR}/picviz/format_ignore_axes.csv.in ${TESTS_FILES_DIR}/picviz/format_ignore_axes.csv.out ${TESTS_FILES_DIR}/picviz/format_ignore_axes.csv.format)
add_test(Trush_format_ignore_axes_url Trush_format_import_export ${TESTS_FILES_DIR}/picviz/format_ignore_axes.url.in ${TESTS_FILES_DIR}/picviz/format_ignore_axes.url.out ${TESTS_FILES_DIR}/picviz/format_ignore_axes.url.format)

# Format ignore rows
add_test(Trush_format_ignore_header Trush_format_import_export ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.in ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.out ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.format)

add_test(Trush_format_ignore_rows_big_header Trush_format_import_export ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.in ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.out ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.format 5)

add_test(Trush_format_ignore_rows_small_header Trush_format_import_export ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.in ${TESTS_FILES_DIR}/picviz/format_ignore_rows_bigh.csv.out ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.format 15)

add_test(Trush_format_ignore_end Trush_format_import_export ${TESTS_FILES_DIR}/picviz/format_ignore_end.csv.in ${TESTS_FILES_DIR}/picviz/format_ignore_end.csv.out ${TESTS_FILES_DIR}/picviz/format_ignore_end.csv.format)

add_test(Trush_format_ignore_header_multifile Trush_format_import_export ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.in ${TESTS_FILES_DIR}/picviz/format_ignore_rows_multifile.csv.out ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.format 15 ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.in)

add_pv_compile_test(Trush_format_upgrade LINKS ${LINK_LIBRARIES} FILES format_upgrade.cpp)
add_test(Trush_format_upgrade Trush_format_upgrade)
# We need to set this env variable as we dump QDom which use a QHash. Without a seed, printing is random.
set_tests_properties(Trush_format_upgrade PROPERTIES ENVIRONMENT "QT_HASH_SEED=0")

add_pv_compile_test(Trush_filter_guess_csv LINKS ${LINK_LIBRARIES} FILES filter_guess_csv.cpp common_guess.cpp)
add_test(Trush_filter_guess_csv_correct Trush_filter_guess_csv 4 "," "\"" ${PVRUSH_TEST_FILES}/csv/tiny.csv)
add_test(Trush_filter_guess_csv_incorrect Trush_filter_guess_csv 0 "," "\"" ${PVRUSH_TEST_FILES}/csv/irregular.csv)
