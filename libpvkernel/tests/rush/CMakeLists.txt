#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

set(LINK_LIBRARIES pvkernel ${QT_LIBRARIES} ${TBB_LIBRARIES} ${ICU_LIBRARY})

#add_definitions(-DPV_USE_RINGBUFFER)

# Test UTF16 conversion
declare_pv_test(Trush_conv_utf16 "conv_utf16.cpp;helpers.cpp" "${LINK_LIBRARIES}")

add_test(TestConvUtf16_utf8 test_conv.sh utf8)
add_test(TestConvUtf16_utf16 test_conv.sh utf16)
add_test(TestConvUtf16_utf32 test_conv.sh utf32)
add_test(TestConvUtf16_latin1 test_conv.sh latin1)

# Test UTF16 conversion and alignement on a newline (PVUnicodeSource object)
declare_pv_test(Trush_conv_utf16_align "conv_utf16_align.cpp;helpers.cpp" "${LINK_LIBRARIES}")

add_test(TestConvUtf16Align diff_stdout.py test-files/conv_utf16_align.out test-files/conv_utf16_align.diff ./Trush_conv_utf16_align test-files/conv_utf16_align 20000)
add_test(TestConvUtf16Align_smallchunksize diff_stdout.py test-files/conv_utf16_align.out test-files/conv_utf16_align.smallchunk.diff ./Trush_conv_utf16_align test-files/conv_utf16_align 2000)

# Test the aggregator
declare_pv_test(Trush_agg "agg.cpp;helpers.cpp" "${LINK_LIBRARIES}")
add_test(TestAggregator diff_stdout.py test-files/aggregator/set1.out test-files/aggregator/set1.diff ./Trush_agg 6000 test-files/aggregator/set1)
add_test(TestAggregator diff_stdout.py test-files/aggregator/set2.out test-files/aggregator/set2.diff ./Trush_agg 6000 test-files/aggregator/set2)

# Test the aggregator in strict mode
declare_pv_test(Trush_agg_strict "agg_strict.cpp;helpers.cpp" "${LINK_LIBRARIES}")
add_test(TestAggregatorStrict diff_stdout.py test-files/aggregator/set1.strict.out test-files/aggregator/set1.strict.diff ./Trush_agg_strict 6000 test-files/aggregator/set1)
add_test(TestAggregatorStrict diff_stdout.py test-files/aggregator/set2.strict.out test-files/aggregator/set2.strict.diff ./Trush_agg_strict 6000 test-files/aggregator/set2)

# Test the controller

# Test the extractor

# Filters tests
declare_pv_test(Trush_splitter_regexp "splitter_regexp.cpp;helpers.cpp" "${LINK_LIBRARIES}")
add_test(TestSplitterRegexp test_splitter_regexp.sh)

declare_pv_test(Trush_splitter_csv "splitter_csv.cpp;helpers.cpp" "${LINK_LIBRARIES}")
add_test(TestSplitterCSV test_splitter_csv.sh)

declare_pv_test(Trush_splitter_url "splitter_url.cpp;helpers.cpp" "${LINK_LIBRARIES}")
add_test(TestSplitterURL test_splitter_url.sh)

declare_pv_test(Trush_splitter_pcap "splitter_pcap.cpp;helpers.cpp" "${LINK_LIBRARIES}")
add_test(TestSplitterPCAP test_splitter_pcap.sh)

declare_pv_test(Trush_grep_regexp "grep_regexp.cpp;helpers.cpp" "${LINK_LIBRARIES}")
add_test(TestGrepRegexp test_grep_regexp.sh)

declare_pv_test(Trush_filter_composition "filter_composition.cpp;helpers.cpp" "${LINK_LIBRARIES}")
add_test(TestFilterComposition diff_stdout.py test-files/filter_composition.out test-files/filter_composition.diff ./Trush_filter_composition test-files/filter_composition 6000)

# Plugin input tests

# Format parsing
declare_pv_test(Trush_format "format.cpp;helpers.cpp" "${LINK_LIBRARIES}")
add_test(TestFormats Trush_format test-files/formats)

declare_pv_test(Trush_format_dump "format_dump.cpp;helpers.cpp" "${LINK_LIBRARIES}")

# For disk file, using source plugins and process through the extractor
# Show the output NRAW as a CSV file
# Used to test the whole PVRush processing w/ files that had some issues with their processing ("had_issues" directory)
declare_pv_test(Trush_process_file "process_file.cpp;helpers.cpp" "${LINK_LIBRARIES}")
add_test(TestHadIssues ./test_issues.sh)

# Auto-discovery test
declare_pv_test(Trush_process_file_auto_discovery "process_file_auto_discovery.cpp;helpers.cpp" "${LINK_LIBRARIES}")

# Test cases for tickets
declare_pv_test(Trush_ticket_1 "ticket_1.cpp" "${LINK_LIBRARIES}")
add_test(TestTicket1 ./Trush_ticket_1)

declare_pv_test(Trush_ticket_2 "ticket_2.cpp" "${LINK_LIBRARIES}")
add_test(TestTicket2 ./Trush_ticket_2)

declare_pv_test(Trush_ticket_28 "ticket_28.cpp" "${LINK_LIBRARIES}")
add_test(TestTicket28 ./Trush_ticket_28)

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
add_test(TestParser Trush_parser)

# Format conversion
declare_pv_test(Trush_format_conv "format_conv.cpp" "${LINK_LIBRARIES}")

# NRAW backend
declare_pv_test(Trush_nraw_backend "nraw_backend.cpp" "${LINK_LIBRARIES}")

# NRAW sort-uniq
declare_pv_test(Trush_nraw_sort_uniq "nraw_sort_uniq.cpp" "${LINK_LIBRARIES}")

if(PYTHONLIBS_FOUND)
	# Python NRAW
	#	add_executable(Trush_python_nraw python_nraw.cpp)
	#target_link_libraries(Trush_python_nraw ${LINK_LIBRARIES} ${PYTHON_LIBRARY} ${Boost_LIBRARIES_PYTHON})
endif(PYTHONLIBS_FOUND)
