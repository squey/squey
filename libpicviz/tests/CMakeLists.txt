#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

configure_file(test_issues.sh.cmake test_issues.sh @ONLY)

set(LINK_LIBRARIES pvkernel picviz ${QT_LIBRARIES} ${TBB_LIBRARIES} ${ICU_LIBRARY} ${OpenMP_LIBRARIES} ${OPENGL_LIBRARIES})

enable_testing()

add_executable(Tpicviz_root root.cpp)
target_link_libraries(Tpicviz_root ${LINK_LIBRARIES})
add_test(TestRoot Tpicviz_root)

#add_executable(Tpicviz_bench_mapping_time bench_mapping-time.cpp)
#target_link_libraries(Tpicviz_bench_mapping_time ${LINK_LIBRARIES})

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

# Source serialization
add_executable(Tpicviz_source_serialize source-serialize.cpp)
target_link_libraries(Tpicviz_source_serialize ${LINK_LIBRARIES})

# Source serialization
add_executable(Tpicviz_root_serialize root-serialize.cpp)
target_link_libraries(Tpicviz_root_serialize ${LINK_LIBRARIES})

# Object sizes
add_executable(Tpicviz_sizes sizes.cpp)
target_link_libraries(Tpicviz_sizes ${LINK_LIBRARIES})

# PVAD2GView
#add_executable(Tpicviz_ad2gview ad2gview.cpp)
#target_link_libraries(Tpicviz_ad2gview ${LINK_LIBRARIES} ${TULIP_LIBRARY})

add_executable(sel_red selection_red.cpp)
target_link_libraries(sel_red ${LINK_LIBRARIES} ${TULIP_LIBRARY})

# Sparse selection
add_executable(Tpicviz_sparse_sel sparse_selection.cpp)
target_link_libraries(Tpicviz_sparse_sel ${LINK_LIBRARIES})

# Forward declaration
add_executable(Tpicviz_forward_decl forward_decl.cpp)
target_link_libraries(Tpicviz_forward_decl ${LINK_LIBRARIES})

# Mapping plugin test
#add_executable(Tpicviz_plugin_mapping plugin-mapping.cpp)
#target_link_libraries(Tpicviz_plugin_mapping ${LINK_LIBRARIES})


# IP mapping performance
#add_executable(Tpicviz_ip_mapping ip-mapping.cpp)
#target_link_libraries(Tpicviz_ip_mapping ${LINK_LIBRARIES})

add_executable(Tpicviz_cat_plotted cat_plotted_file.cpp)
target_link_libraries(Tpicviz_cat_plotted ${LINK_LIBRARIES})

add_executable(Tpicviz_process picviz-process.cpp)
target_link_libraries(Tpicviz_process ${LINK_LIBRARIES})
