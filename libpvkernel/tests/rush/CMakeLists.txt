include_directories(AFTER ${QT_QTCORE_INCLUDE_DIR})
include_directories(AFTER ${QT_INCLUDE_DIR})

set(LINK_LIBRARIES ${PVKERNEL_LIBRARY} ${QT_LIBRARIES} ${TBB_LIBRARIES} ${TBB_MALLOC_LIBRARY} ${ICU_LIBRARY})

#add_definitions(-DPV_USE_RINGBUFFER)

# Test UTF16 conversion
add_executable(Trush_conv_utf16 conv_utf16.cpp helpers.cpp)
target_link_libraries(Trush_conv_utf16 ${LINK_LIBRARIES})

add_test(TestConvUtf16_utf8 test_conv.sh utf8)
add_test(TestConvUtf16_utf16 test_conv.sh utf16)
add_test(TestConvUtf16_utf32 test_conv.sh utf32)
add_test(TestConvUtf16_latin1 test_conv.sh latin1)

# Test UTF16 conversion and alignement on a newline (PVUnicodeSource object)
add_executable(Trush_conv_utf16_align conv_utf16_align.cpp helpers.cpp)
target_link_libraries(Trush_conv_utf16_align ${LINK_LIBRARIES})

add_test(TestConvUtf16Align diff_stdout.py test-files/conv_utf16_align.out test-files/conv_utf16_align.diff ./Trush_conv_utf16_align test-files/conv_utf16_align 20000)
add_test(TestConvUtf16Align_smallchunksize diff_stdout.py test-files/conv_utf16_align.out test-files/conv_utf16_align.smallchunk.diff ./Trush_conv_utf16_align test-files/conv_utf16_align 2000)

# Test the aggregator
add_executable(Trush_agg agg.cpp helpers.cpp)
target_link_libraries(Trush_agg ${LINK_LIBRARIES})
add_test(TestAggregator diff_stdout.py test-files/aggregator/set1.out test-files/aggregator/set1.diff ./Trush_agg 6000 test-files/aggregator/set1)
add_test(TestAggregator diff_stdout.py test-files/aggregator/set2.out test-files/aggregator/set2.diff ./Trush_agg 6000 test-files/aggregator/set2)

# Test the controller

# Test the extractor

# Filters tests
add_executable(Trush_splitter_regexp splitter_regexp.cpp helpers.cpp)
target_link_libraries(Trush_splitter_regexp ${LINK_LIBRARIES})
add_test(TestSplitterRegexp test_splitter_regexp.sh)

add_executable(Trush_splitter_csv splitter_csv.cpp helpers.cpp)
target_link_libraries(Trush_splitter_csv ${LINK_LIBRARIES})
add_test(TestSplitterCSV test_splitter_csv.sh)

add_executable(Trush_splitter_url splitter_url.cpp helpers.cpp)
target_link_libraries(Trush_splitter_url ${LINK_LIBRARIES})
add_test(TestSplitterURL test_splitter_url.sh)

add_executable(Trush_splitter_pcap splitter_pcap.cpp helpers.cpp)
target_link_libraries(Trush_splitter_pcap ${LINK_LIBRARIES})
add_test(TestSplitterPCAP test_splitter_pcap.sh)

add_executable(Trush_grep_regexp grep_regexp.cpp helpers.cpp)
target_link_libraries(Trush_grep_regexp ${LINK_LIBRARIES})
add_test(TestGrepRegexp test_grep_regexp.sh)

add_executable(Trush_filter_composition filter_composition.cpp helpers.cpp)
target_link_libraries(Trush_filter_composition ${LINK_LIBRARIES})
add_test(TestFilterComposition diff_stdout.py test-files/filter_composition.out test-files/filter_composition.diff ./Trush_filter_composition test-files/filter_composition 6000)

# Plugin input tests

# Format parsing
add_executable(Trush_format format.cpp)
target_link_libraries(Trush_format ${LINK_LIBRARIES})
add_test(TestFormats Trush_format test-files/formats)

add_executable(Trush_format_dump format_dump.cpp)
target_link_libraries(Trush_format_dump ${LINK_LIBRARIES})

# For disk file, using source plugins and process through the extractor
# Show the output NRAW as a CSV file
# Used to test the whole PVRush processing w/ files that had some issues with their processing ("had_issues" directory)
add_executable(Trush_process_file process_file.cpp helpers.cpp)
target_link_libraries(Trush_process_file ${LINK_LIBRARIES})
add_test(TestHadIssues ./test_issues.sh)

# Test cases for tickets
add_executable(Trush_ticket_1 ticket_1.cpp)
target_link_libraries(Trush_ticket_1 ${LINK_LIBRARIES})
add_test(TestTicket1 ./Trush_ticket_1)

add_executable(Trush_ticket_2 ticket_2.cpp helpers.cpp)
target_link_libraries(Trush_ticket_2 ${LINK_LIBRARIES})
add_test(TestTicket2 ./Trush_ticket_2)

add_executable(Trush_ticket_28 ticket_28.cpp)
target_link_libraries(Trush_ticket_28 ${LINK_LIBRARIES})
add_test(TestTicket28 ./Trush_ticket_28)

# Benchmark for the whole extractor
add_executable(Trush_bench bench_extractor.cpp)
target_link_libraries(Trush_bench ${LINK_LIBRARIES})
#add_test(TestBench Trush_bench)

add_executable(Trush_input_hdfs input_hdfs.cpp)
target_link_libraries(Trush_input_hdfs ${LINK_LIBRARIES})

# libarchive auto detect
#add_executable(Trush_archive_detect archive_detect.cpp ../plugins/input_types/file/extract.cpp)
#target_link_libraries(Trush_archive_detect ${LINK_LIBRARIES})

# Filter guessing
add_executable(Trush_filter_guess filter_guess.cpp)
target_link_libraries(Trush_filter_guess ${LINK_LIBRARIES})

add_executable(Trush_parser parser.cpp)
target_link_libraries(Trush_parser ${LINK_LIBRARIES})
add_test(TestParser Trush_parser)
