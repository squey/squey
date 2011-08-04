include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpicviz/src/include)

add_definitions(${PICVIZ_DEFINITIONS})
include_directories(AFTER ${PICVIZ_INCLUDE_DIRS})


set(LINK_LIBRARIES ${PICVIZ_LIBRARY} ${DUMBNET_LIBRARIES} ${PVKERNEL_LIBRARY} ${QT_LIBRARIES} ${TBB_LIBRARY} ${TBB_MALLOC_LIBRARY})

enable_testing()

add_executable(Tpicviz_root root.cpp)
target_link_libraries(Tpicviz_root ${LINK_LIBRARIES})
add_test(TestRoot Tpicviz_root)

add_executable(Tpicviz_bench_mapping_time bench_mapping-time.cpp)
target_link_libraries(Tpicviz_bench_mapping_time ${LINK_LIBRARIES})


add_executable(TPVSelection PVSelection.cpp)
target_link_libraries(TPVSelection ${LINK_LIBRARIES})
add_test(TestPVSelection TPVSelection)

# Mapping testing (had_issues/mapping)
add_executable(Tpicviz_process_file_mapping process_file_mapping.cpp)
target_link_libraries(Tpicviz_process_file_mapping ${LINK_LIBRARIES})
add_test(TestIssuesMapping test_issues.sh mapping)

# Plotting testing (had_issues/plotting)
add_executable(Tpicviz_process_file_plotting process_file_plotting.cpp)
target_link_libraries(Tpicviz_process_file_plotting ${LINK_LIBRARIES})
add_test(TestIssuesPlotting test_issues.sh plotting)

# Mapping plugin test
#add_executable(Tpicviz_plugin_mapping plugin-mapping.cpp)
#target_link_libraries(Tpicviz_plugin_mapping ${LINK_LIBRARIES})


# IP mapping performance
#add_executable(Tpicviz_ip_mapping ip-mapping.cpp)
#target_link_libraries(Tpicviz_ip_mapping ${LINK_LIBRARIES})
