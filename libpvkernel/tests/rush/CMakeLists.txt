#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

configure_file(test_issues.sh.cmake test_issues.sh @ONLY)
configure_file(test_splitter_csv.sh.cmake test_splitter_csv.sh @ONLY)
configure_file(test_splitter_duplicate.sh.cmake test_splitter_duplicate.sh @ONLY)
configure_file(test_splitter_pcap.sh.cmake test_splitter_pcap.sh @ONLY)
configure_file(test_splitter_regexp.sh.cmake test_splitter_regexp.sh @ONLY)
configure_file(test_splitter_url.sh.cmake test_splitter_url.sh @ONLY)
configure_file(test_grep_regexp.sh.cmake test_grep_regexp.sh @ONLY)

set(LINK_LIBRARIES pvkernel ${QT_LIBRARIES} ${TBB_LIBRARIES} ${ICU_LIBRARY})

# Temporary output directory for NRAW disk backend based tests
if (NOT DEFINED PVKERNEL_RUSH_TEST_NRAW_TMP)
	message(WARNING "PVKERNEL_RUSH_TEST_NRAW_TMP not defined, /tmp will be used by default")
	set(PVKERNEL_RUSH_TEST_NRAW_TMP "/tmp")
endif()

#add_definitions(-DPV_USE_RINGBUFFER)

set(PVRUSH_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/test-files")
set(PVRUSH_TEST_FILES_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/test-files")

# Test UTF16 conversion
declare_pv_test(Trush_conv_utf16 "conv_utf16.cpp;helpers.cpp" "${LINK_LIBRARIES}")

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
declare_pv_test(Trush_conv_utf16_align "conv_utf16_align.cpp;helpers.cpp" "${LINK_LIBRARIES}")

macro(PV_ADD_TEST_CONV_UTF16_ALIGN CASE CHUNK_SIZE)
	add_test(TestConvUtf16Align_${CASE}_${CHUNK_SIZE} ${TEST_DIFF_STDOUT}
		${PVRUSH_TEST_FILES}/${CASE}.out
		${PVRUSH_TEST_FILES_OUTPUT}/${CASE}.diff
		${CMAKE_CURRENT_BINARY_DIR}/Trush_conv_utf16_align
		${PVRUSH_TEST_FILES}/${CASE} ${CHUNK_SIZE})
endmacro()

PV_ADD_TEST_CONV_UTF16_ALIGN(conv_utf16_align 20000)
PV_ADD_TEST_CONV_UTF16_ALIGN(conv_utf16_align 2000)
PV_ADD_TEST_CONV_UTF16_ALIGN(conv_utf16_align_newline 20000)
PV_ADD_TEST_CONV_UTF16_ALIGN(conv_utf16_align_newline 2000)

# Test the aggregator
declare_pv_test(Trush_agg "agg.cpp;helpers.cpp" "${LINK_LIBRARIES}")
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
declare_pv_test(Trush_agg_strict "agg_strict.cpp;helpers.cpp" "${LINK_LIBRARIES}")
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

# Test the controller

# Test the extractor

declare_pv_test(Trush_splitter_regexp "splitter_regexp.cpp;helpers.cpp" "${LINK_LIBRARIES}")
add_test(TestSplitterRegexp ${CMAKE_CURRENT_BINARY_DIR}/test_splitter_regexp.sh)

declare_pv_test(Trush_splitter_csv "splitter_csv.cpp;helpers.cpp" "${LINK_LIBRARIES}")
add_test(TestSplitterCSV ${CMAKE_CURRENT_BINARY_DIR}/test_splitter_csv.sh)

declare_pv_test(Trush_splitter_url "splitter_url.cpp;helpers.cpp" "${LINK_LIBRARIES}")
add_test(TestSplitterURL ${CMAKE_CURRENT_BINARY_DIR}/test_splitter_url.sh)

declare_pv_test(Trush_splitter_pcap "splitter_pcap.cpp;helpers.cpp" "${LINK_LIBRARIES}")
add_test(TestSplitterPCAP ${CMAKE_CURRENT_BINARY_DIR}/test_splitter_pcap.sh)

declare_pv_test(Trush_grep_regexp "grep_regexp.cpp;helpers.cpp" "${LINK_LIBRARIES}")
add_test(TestGrepRegexp ${CMAKE_CURRENT_BINARY_DIR}/test_grep_regexp.sh)

declare_pv_test(Trush_splitter_duplicate "splitter_duplicate.cpp;helpers.cpp" "${LINK_LIBRARIES}")
add_test(TestSplitterDuplicate ${CMAKE_CURRENT_BINARY_DIR}/test_splitter_duplicate.sh)

declare_pv_test(Trush_filter_composition "filter_composition.cpp;helpers.cpp" "${LINK_LIBRARIES}")
add_test(TestFilterComposition ${TEST_DIFF_STDOUT}
	${PVRUSH_TEST_FILES}/filter_composition.out
	${PVRUSH_TEST_FILES}/filter_composition.diff
	${CMAKE_CURRENT_BINARY_DIR}/Trush_filter_composition
	${PVRUSH_TEST_FILES}/filter_composition 6000)

# Plugin input tests

# Format parsing
declare_pv_test(Trush_format "format.cpp;helpers.cpp" "${LINK_LIBRARIES}")
add_test(TestFormats Trush_format ${PVRUSH_TEST_FILES}/formats)

declare_pv_test(Trush_format_dump "format_dump.cpp;helpers.cpp" "${LINK_LIBRARIES}")

# For disk file, using source plugins and process through the extractor
# Show the output NRAW as a CSV file
# Used to test the whole PVRush processing w/ files that had some issues with their processing ("had_issues" directory)
# AG: this will be done with a better test method that test every elements of the processing pipeline
#declare_pv_test(Trush_process_file "process_file.cpp;helpers.cpp" "${LINK_LIBRARIES}")
#add_test(TestHadIssues ${CMAKE_CURRENT_SOURCE_DIR}/test_issues.sh)

# Auto-discovery test
declare_pv_test(Trush_process_file_auto_discovery "process_file_auto_discovery.cpp;helpers.cpp" "${LINK_LIBRARIES}")

# Test cases for tickets
declare_pv_test(Trush_ticket_1 "ticket_1.cpp" "${LINK_LIBRARIES}")
add_test(TestTicket1 Trush_ticket_1 ${PVRUSH_TEST_FILES})

declare_pv_test(Trush_ticket_2 "ticket_2.cpp" "${LINK_LIBRARIES}")
add_test(TestTicket2 Trush_ticket_2)

declare_pv_test(Trush_ticket_28 "ticket_28.cpp" "${LINK_LIBRARIES}")
add_test(TestTicket28 Trush_ticket_28 ${PVRUSH_TEST_FILES})

# Benchmark for the whole extractor
#add_executable(Trush_bench bench_extractor.cpp)
#target_link_libraries(Trush_bench ${LINK_LIBRARIES})
#add_test(TestBench Trush_bench)

# Memory footprint benchmark
declare_pv_test(Trush_mem "bench_memory.cpp" "${LINK_LIBRARIES}")

declare_pv_test(Trush_input_hdfs "input_hdfs.cpp" "${LINK_LIBRARIES}")

if(HDFS_FOUND)
	#add_executable(Trush_input_hadoop input_hadoop.cpp helpers.cpp ../../plugins/common/hdfs/PVInputHDFSFile.cpp ../../plugins/common/hdfs/PVInputHDFSServer.cpp)
	#target_link_libraries(Trush_input_hadoop ${LINK_LIBRARIES} ${HDFS_LIBRARIES})
endif(HDFS_FOUND)

declare_pv_test(Trush_fake_hadoop_src "fake_hadoop_src.cpp" "${LINK_LIBRARIES}")

# libarchive auto detect
#add_executable(Trush_archive_detect archive_detect.cpp ../plugins/input_types/file/extract.cpp)
#target_link_libraries(Trush_archive_detect ${LINK_LIBRARIES})

# Filter guessing
declare_pv_test(Trush_filter_guess "filter_guess.cpp" "${LINK_LIBRARIES}")
declare_pv_test(Trush_parser "parser.cpp" "${LINK_LIBRARIES}")

# Format conversion
declare_pv_test(Trush_format_conv "format_conv.cpp" "${LINK_LIBRARIES}")

# NRAW backend
declare_pv_test(Trush_nraw_backend "nraw_backend.cpp" "${LINK_LIBRARIES}")

# NRAW sort-uniq
declare_pv_test(Trush_nraw_sort_uniq "nraw_sort_uniq.cpp" "${LINK_LIBRARIES}")

# NRAW sort-column
declare_pv_test(Trush_nraw_sort_column "nraw_sort_column.cpp" "${LINK_LIBRARIES}")

# NRAW create
declare_pv_test(Trush_nraw_create "nraw_create.cpp;helpers.cpp" "${LINK_LIBRARIES}")

# NRAW load bench
declare_pv_test(Trush_nraw_load_bench "nraw_load_bench.cpp" "${LINK_LIBRARIES}")

# NRAW integrity test
declare_pv_test(Trush_nraw_integrity_test "nraw_integrity_test.cpp" "${LINK_LIBRARIES}")
add_test(TestNrawIntegrity Trush_nraw_integrity_test ${PVKERNEL_RUSH_TEST_NRAW_TMP})

declare_pv_test(arcsight_src "test_arcsight_source.cpp;helpers.cpp;${CMAKE_SOURCE_DIR}/libpvkernel/plugins/common/arcsight/PVArcsightInfos.cpp;${CMAKE_SOURCE_DIR}/libpvkernel/plugins/common/arcsight/PVArcsightQuery.cpp;" "${LINK_LIBRARIES}")

if(PYTHONLIBS_FOUND)
	# Python NRAW
	#	add_executable(Trush_python_nraw python_nraw.cpp)
	#target_link_libraries(Trush_python_nraw ${LINK_LIBRARIES} ${PYTHON_LIBRARY} ${Boost_LIBRARIES_PYTHON})
endif(PYTHONLIBS_FOUND)
