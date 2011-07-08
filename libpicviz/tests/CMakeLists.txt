include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpicviz/src/include)

add_definitions(${PICVIZ_DEFINITIONS})
include_directories(AFTER ${PICVIZ_INCLUDE_DIRS})


if(WIN32)
	set(PICVIZ_LIBRARY picviz)
else(WIN32)
	set(PICVIZ_LIBRARY picviz m)
endif(WIN32)

set(LINK_LIBRARIES ${APR_LIBS} ${PCRE_PCRE_LIBRARY} ${PCAP_LIBRARIES} ${DUMBNET_LIBRARIES} ${PICVIZ_LIBRARY} ${PVCORE_LIBRARY} ${PVRUSH_LIBRARY} ${QT_LIBRARIES} ${TBB_LIBRARY} -lrt)

enable_testing()

add_executable(Tpicviz_root root.cpp)
target_link_libraries(Tpicviz_root ${LINK_LIBRARIES})
add_test(TestRoot Tpicviz_root)

add_executable(Tpicviz_bench_mapping_time bench_mapping-time.cpp)
target_link_libraries(Tpicviz_bench_mapping_time ${LINK_LIBRARIES})


add_executable(TPVSelection PVSelection.cpp)
target_link_libraries(TPVSelection ${LINK_LIBRARIES})
add_test(TestPVSelection TPVSelection)

# Map testing (had_issues/mapping)
add_executable(Tpicviz_process_file_mapping process_file_mapping.cpp)
target_link_libraries(Tpicviz_process_file_mapping ${LINK_LIBRARIES})
add_test(TestIssuesMapping test_issues.sh mapping)
