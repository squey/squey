#
# @file
#

set(PVRUSH_TEST_FILES "${TESTS_FILES_DIR}/pvkernel/rush")
set(PVRUSH_TEST_FILES_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/test-files")

# Test UTF8 conversion
add_squey_compile_test(Trush_conv_utf8 LINKS ${LINK_LIBRARIES} FILES conv_utf8.cpp helpers.cpp)
add_bench(Trush_conv_utf8 LINKS ${LINK_LIBRARIES} FILES conv_utf8.cpp helpers.cpp)

macro(PV_ADD_TEST_CONV_UTF8 input_file chunks_size)
	add_squey_test(TestConvUtf8_${input_file}_${chunks_size} Trush_conv_utf8 ${PVRUSH_TEST_FILES}/${input_file} ${PVRUSH_TEST_FILES}/${input_file}.out ${chunks_size})
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
PV_ADD_TEST_CONV_UTF8("charset/viscii" 10048576)

# Test the aggregator
add_squey_compile_test(Trush_agg LINKS ${LINK_LIBRARIES} FILES agg.cpp helpers.cpp)
add_squey_test(TestAggregator1 Trush_agg 6000 ${PVRUSH_TEST_FILES}/aggregator/set1)
add_squey_test(TestAggregator2 Trush_agg 6000 ${PVRUSH_TEST_FILES}/aggregator/set2)
add_squey_test(TestAggregator3 Trush_agg 6000 ${PVRUSH_TEST_FILES}/aggregator/set3)

# Test for filters
add_squey_compile_test(Trush_filter_grep LINKS ${LINK_LIBRARIES} FILES filter_grep.cpp helpers.cpp)
add_squey_test(Trush_filter_grep Trush_filter_grep)
add_bench(Trush_filter_grep LINKS ${LINK_LIBRARIES} FILES filter_grep.cpp helpers.cpp)

# Test for converter
add_squey_compile_test(Trush_converter_GUID_to_IPV4 LINKS ${LINK_LIBRARIES} FILES converter_GUID_to_IPV4.cpp helpers.cpp)
add_squey_test(Trush_converter_GUID_to_IPV4 Trush_converter_GUID_to_IPV4)
add_bench(Trush_converter_GUID_to_IPV4 LINKS ${LINK_LIBRARIES} FILES converter_GUID_to_IPV4.cpp helpers.cpp)

add_squey_compile_test(Trush_converter_GUID_to_IPV6 LINKS ${LINK_LIBRARIES} FILES converter_GUID_to_IPV6.cpp helpers.cpp)
add_squey_test(Trush_converter_GUID_to_IPV6 Trush_converter_GUID_to_IPV6)
add_bench(Trush_converter_GUID_to_IPV6 LINKS ${LINK_LIBRARIES} FILES converter_GUID_to_IPV6.cpp helpers.cpp)

add_squey_compile_test(Trush_converter_substitution LINKS ${LINK_LIBRARIES} FILES converter_substitution.cpp helpers.cpp)
add_squey_test(Trush_converter_substitution Trush_converter_substitution)
add_bench(Trush_converter_substitution LINKS ${LINK_LIBRARIES} FILES converter_substitution.cpp helpers.cpp)

add_squey_compile_test(Trush_converter_struct LINKS ${LINK_LIBRARIES} FILES converter_struct.cpp helpers.cpp)
add_squey_test(Trush_converter_struct Trush_converter_struct)
add_bench(Trush_converter_struct LINKS ${LINK_LIBRARIES} FILES converter_struct.cpp helpers.cpp)

# Test for splitters
add_squey_compile_test(Trush_splitter_key_value LINKS ${LINK_LIBRARIES} FILES splitter_key_value.cpp helpers.cpp)
add_squey_test(Trush_splitter_key_value Trush_splitter_key_value)
add_bench(Trush_splitter_key_value LINKS ${LINK_LIBRARIES} FILES splitter_key_value.cpp helpers.cpp)

add_squey_compile_test(Trush_splitter_dns_fqdn LINKS ${LINK_LIBRARIES} FILES splitter_dns_fqdn.cpp helpers.cpp)
add_squey_test(Trush_splitter_dns_fqdn Trush_splitter_dns_fqdn)
add_bench(Trush_splitter_dns_fqdn LINKS ${LINK_LIBRARIES} FILES splitter_dns_fqdn.cpp helpers.cpp)

add_squey_compile_test(Trush_splitter_mac LINKS ${LINK_LIBRARIES} FILES splitter_mac.cpp helpers.cpp)
add_squey_test(Trush_splitter_mac Trush_splitter_mac)
add_bench(Trush_splitter_mac LINKS ${LINK_LIBRARIES} FILES splitter_mac.cpp helpers.cpp)

add_squey_compile_test(Trush_splitter_ipv4 LINKS ${LINK_LIBRARIES} FILES splitter_ipv4.cpp helpers.cpp)
add_squey_test(Trush_splitter_ipv4 Trush_splitter_ipv4)
add_bench(Trush_splitter_ipv4 LINKS ${LINK_LIBRARIES} FILES splitter_ipv4.cpp helpers.cpp)

add_squey_compile_test(Trush_splitter_ipv6 LINKS ${LINK_LIBRARIES} FILES splitter_ipv6.cpp helpers.cpp)
add_squey_test(Trush_splitter_ipv6 Trush_splitter_ipv6)
add_bench(Trush_splitter_ipv6 LINKS ${LINK_LIBRARIES} FILES splitter_ipv6.cpp helpers.cpp)

add_squey_compile_test(Trush_splitter_regexp LINKS ${LINK_LIBRARIES} FILES splitter_regexp.cpp helpers.cpp)
add_squey_test(Trush_splitter_regexp Trush_splitter_regexp)
add_bench(Trush_splitter_regexp LINKS ${LINK_LIBRARIES} FILES splitter_regexp.cpp helpers.cpp)

add_squey_compile_test(Trush_splitter_regexp_full_line LINKS ${LINK_LIBRARIES} FILES splitter_regexp_full_line.cpp helpers.cpp)
add_squey_test(Trush_splitter_regexp_full_line Trush_splitter_regexp_full_line)
add_bench(Trush_splitter_regexp_full_line LINKS ${LINK_LIBRARIES} FILES splitter_regexp_full_line.cpp helpers.cpp)

add_squey_compile_test(Trush_splitter_csv LINKS ${LINK_LIBRARIES} FILES splitter_csv.cpp helpers.cpp)
add_squey_test(Trush_splitter_csv Trush_splitter_csv)
add_bench(Trush_splitter_csv LINKS ${LINK_LIBRARIES} FILES splitter_csv.cpp helpers.cpp)

add_squey_test(Trush_splitter_csv2 Trush_splitter_csv ${PVRUSH_TEST_FILES}/splitters/csv/file.csv ${PVRUSH_TEST_FILES}/splitters/csv/file.csv.out 7)
add_squey_test(Trush_splitter_csv3 Trush_splitter_csv ${PVRUSH_TEST_FILES}/splitters/csv/splitter_csv_escaped.csv ${PVRUSH_TEST_FILES}/splitters/csv/splitter_csv_escaped.csv.out 2)
add_squey_test(Trush_splitter_csv4 Trush_splitter_csv ${PVRUSH_TEST_FILES}/splitters/csv/splitter_csv_doubled_quote.csv ${PVRUSH_TEST_FILES}/splitters/csv/splitter_csv_doubled_quote.csv.out 20)

add_squey_compile_test(Trush_splitter_csv_nested LINKS ${LINK_LIBRARIES} FILES splitter_csv_nested.cpp helpers.cpp)
add_squey_test(Trush_splitter_csv_nested Trush_splitter_csv_nested)

add_squey_compile_test(Trush_splitter_url LINKS ${LINK_LIBRARIES} FILES splitter_url.cpp helpers.cpp)
add_squey_test(Trush_splitter_url Trush_splitter_url)
add_bench(Trush_splitter_url LINKS ${LINK_LIBRARIES} FILES splitter_url.cpp helpers.cpp)

add_squey_compile_test(Trush_splitter_duplicate LINKS ${LINK_LIBRARIES} FILES splitter_duplicate.cpp helpers.cpp)
add_squey_test(Trush_splitter_duplicate Trush_splitter_duplicate)
add_bench(Trush_splitter_duplicate LINKS ${LINK_LIBRARIES} FILES splitter_duplicate.cpp helpers.cpp)

add_squey_compile_test(Trush_splitter_length LINKS ${LINK_LIBRARIES} FILES splitter_length.cpp helpers.cpp)
add_squey_test(Trush_splitter_length Trush_splitter_length)

add_squey_compile_test(Trush_filter_composition LINKS ${LINK_LIBRARIES} FILES filter_composition.cpp helpers.cpp)
add_squey_test(Trush_filter_composition Trush_filter_composition)
add_bench(Trush_filter_composition LINKS ${LINK_LIBRARIES} FILES filter_composition.cpp helpers.cpp)

# Plugin database
find_package(Qt6Sql REQUIRED)
include_directories(SYSTEM ${Qt6Sql_INCLUDE_DIRS})
set(DATABASE_SRC_FILES
../../plugins/common/database/PVDBInfos.cpp
../../plugins/common/database/PVDBQuery.cpp
../../plugins/common/database/PVSQLTypeMap.cpp
helpers.cpp)

add_squey_compile_test(Trush_database LINKS ${LINK_LIBRARIES} pvkernel Qt6Sql FILES database.cpp ${DATABASE_SRC_FILES})
#add_squey_test(Trush_database Trush_database ${TESTS_FILES_DIR})

# Plugin sources

add_squey_compile_test(Trush_elasticsearch_source LINKS ${LINK_LIBRARIES}
	FILES elasticsearch_source.cpp helpers.cpp
	../../plugins/common/elasticsearch/PVElasticsearchInfos.cpp
	../../plugins/common/elasticsearch/PVElasticsearchQuery.cpp)
#add_squey_test(Trush_elasticsearch_source Trush_elasticsearch_source)
add_bench(Trush_elasticsearch_source LINKS ${LINK_LIBRARIES}
	FILES elasticsearch_source.cpp helpers.cpp
	../../plugins/common/elasticsearch/PVElasticsearchInfos.cpp
	../../plugins/common/elasticsearch/PVElasticsearchQuery.cpp)

# Plugin input tests
add_squey_compile_test(Trush_nraw_filtered LINKS ${LINK_LIBRARIES} FILES nraw_filtered.cpp)
add_squey_test(Trush_nraw_filtered Trush_nraw_filtered)
add_bench(Trush_nraw_filtered LINKS ${LINK_LIBRARIES} FILES nraw_filtered.cpp)

# Check NRaw fill and dump
add_squey_compile_test(Trush_bad_splitter LINKS ${LINK_LIBRARIES} FILES bad_splitter.cpp)
add_squey_test(Trush_bad_splitter Trush_bad_splitter)
add_bench(Trush_bad_splitter LINKS ${LINK_LIBRARIES} FILES bad_splitter.cpp)

add_squey_compile_test(Trush_nraw_create LINKS ${LINK_LIBRARIES} FILES nraw_create.cpp)
add_squey_test(Trush_nraw_create Trush_nraw_create)
add_bench(Trush_nraw_create LINKS ${LINK_LIBRARIES} FILES nraw_create.cpp)

add_squey_compile_test(Trush_nraw_dump LINKS ${LINK_LIBRARIES} FILES nraw_dump.cpp)
add_squey_test(Trush_nraw_dump Trush_nraw_dump)
add_bench(Trush_nraw_dump LINKS ${LINK_LIBRARIES} FILES nraw_dump.cpp)

add_squey_compile_test(Trush_format LINKS ${LINK_LIBRARIES} FILES format.cpp helpers.cpp)
add_squey_test(Trush_format Trush_format ${PVRUSH_TEST_FILES}/formats)

add_squey_compile_test(Trush_ticket_1 LINKS ${LINK_LIBRARIES} FILES ticket_1.cpp)
add_squey_test(Trush_ticket_1 Trush_ticket_1 ${PVRUSH_TEST_FILES})

add_squey_compile_test(Trush_ticket_2 LINKS ${LINK_LIBRARIES} FILES ticket_2.cpp)
add_squey_test(LONG_Trush_ticket_2 Trush_ticket_2)

add_squey_compile_test(Trush_ticket_28 LINKS ${LINK_LIBRARIES} FILES ticket_28.cpp)
add_squey_test(Trush_ticket_28 Trush_ticket_28 ${PVRUSH_TEST_FILES})

add_squey_compile_test(Trush_agg_index_with_grow LINKS ${LINK_LIBRARIES} FILES agg_index_grow.cpp)
add_squey_test(Trush_agg_index_with_grow Trush_agg_index_with_grow)

# Datetime ICU to boost
add_squey_compile_test(Trush_datetime_support LINKS ${LINK_LIBRARIES} FILES datetime_support.cpp)
add_squey_test(Trush_datetime_support Trush_datetime_support)

# Format ignore axes
add_squey_compile_test(Trush_format_import_export LINKS ${LINK_LIBRARIES} FILES format_import_export.cpp)
add_squey_test(Trush_format_ignore_axes_csv Trush_format_import_export ${TESTS_FILES_DIR}/picviz/format_ignore_axes.csv.in ${TESTS_FILES_DIR}/picviz/format_ignore_axes.csv.out ${TESTS_FILES_DIR}/picviz/format_ignore_axes.csv.format)
add_squey_test(Trush_format_ignore_axes_url Trush_format_import_export ${TESTS_FILES_DIR}/picviz/format_ignore_axes.url.in ${TESTS_FILES_DIR}/picviz/format_ignore_axes.url.out ${TESTS_FILES_DIR}/picviz/format_ignore_axes.url.format)

# Format ignore rows
add_squey_test(Trush_format_ignore_header Trush_format_import_export ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.in ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.out ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.format)

add_squey_test(Trush_format_ignore_rows_big_header Trush_format_import_export ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.in ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.out ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.format 5)

add_squey_test(Trush_format_ignore_rows_small_header Trush_format_import_export ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.in ${TESTS_FILES_DIR}/picviz/format_ignore_rows_bigh.csv.out ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.format 15)

add_squey_test(Trush_format_ignore_end Trush_format_import_export ${TESTS_FILES_DIR}/picviz/format_ignore_end.csv.in ${TESTS_FILES_DIR}/picviz/format_ignore_end.csv.out ${TESTS_FILES_DIR}/picviz/format_ignore_end.csv.format)

add_squey_test(Trush_format_ignore_header_multifile Trush_format_import_export ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.in ${TESTS_FILES_DIR}/picviz/format_ignore_rows_multifile.csv.out ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.format 15 ${TESTS_FILES_DIR}/picviz/format_ignore_rows.csv.in)

# Check a csv without enough fields
add_squey_test(Trush_not_enough_column Trush_format_import_export ${TESTS_FILES_DIR}/picviz/test_petit.log null ${TESTS_FILES_DIR}/picviz/apartments.csv.format)

# Check a filtered file with headers
add_squey_test(Trush_filtered_with_header Trush_format_import_export ${TESTS_FILES_DIR}/pvkernel/rush/filter_grep_big ${TESTS_FILES_DIR}/pvkernel/rush/filtered_with_header ${TESTS_FILES_DIR}/pvkernel/rush/filter_grep.format)

add_squey_compile_test(Trush_format_upgrade LINKS ${LINK_LIBRARIES} FILES format_upgrade.cpp)
add_squey_test(Trush_format_upgrade Trush_format_upgrade)

add_squey_compile_test(Trush_filter_guess_csv LINKS ${LINK_LIBRARIES} FILES filter_guess_csv.cpp)
add_squey_test(Trush_filter_guess_csv_correct Trush_filter_guess_csv 4 "," "\"" ${PVRUSH_TEST_FILES}/csv/tiny.csv)
add_squey_test(Trush_filter_guess_csv_incorrect Trush_filter_guess_csv 0 "," "\"" ${PVRUSH_TEST_FILES}/csv/irregular.csv)

add_squey_compile_test(Trush_axes_types_discovery LINKS ${LINK_LIBRARIES} FILES axes_types_discovery.cpp)
add_squey_test(Trush_axes_types_discovery Trush_axes_types_discovery)
add_bench(Trush_axes_types_discovery LINKS ${LINK_LIBRARIES} FILES axes_types_discovery.cpp)

# squey-logstats
add_squey_compile_test(Trush_logstats LINKS ${LINK_LIBRARIES} FILES log_stats.cpp)
add_squey_test(Trush_logstats Trush_logstats)

set(PARQUET_SRC_FILES
../../plugins/common/parquet/PVParquetAPI.cpp
../../plugins/input-types/parquet/PVParquetExporter.cpp
)

add_squey_compile_test(Trush_apache_parquet LINKS ${LINK_LIBRARIES} ${PARQUET_LIBS} FILES apache_parquet.cpp ${PARQUET_SRC_FILES})
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
target_compile_options(SQUEY_TEST_Trush_apache_parquet PRIVATE "-O0") # Optimisations are breaking the test under MacOS
endif()
set(PARQUET_MULTIINPUTS_CSV "${TESTS_FILES_DIR}/pvkernel/rush/parquet/parquet_multïinputs.csv")
set(PARQUET_MULTIINPUTS_OPTIM_CSV "${TESTS_FILES_DIR}/pvkernel/rush/parquet/parquet_multïinputs_optim.csv")
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	set(PARQUET_MULTIINPUTS_CSV "${PARQUET_MULTIINPUTS_CSV}.mac")
	set(PARQUET_MULTIINPUTS_OPTIM_CSV "${PARQUET_MULTIINPUTS_OPTIM_CSV}.mac")
endif()
add_squey_test(Trush_apache_parquet Trush_apache_parquet ${PARQUET_MULTIINPUTS_CSV} ${PARQUET_MULTIINPUTS_OPTIM_CSV} ${TESTS_FILES_DIR}/pvkernel/rush/parquet/multïchunks.parquet ${TESTS_FILES_DIR}/pvkernel/rush/parquet/parquet_multïchunks_export.csv)
add_squey_compile_test(Trush_apache_parquet_multichunks LINKS ${LINK_LIBRARIES} ${PARQUET_LIBS} FILES apache_parquet_multichunks.cpp ${PARQUET_SRC_FILES})
add_squey_test(Trush_apache_parquet_multichunks Trush_apache_parquet_multichunks ${TESTS_FILES_DIR}/pvkernel/rush/parquet/multïchunks.parquet ${TESTS_FILES_DIR}/pvkernel/rush/parquet/parquet_multïchunks.csv)

file(RELATIVE_PATH REL_TEST_DIR ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake DESTINATION "tests/${REL_TEST_DIR}" COMPONENT tests)