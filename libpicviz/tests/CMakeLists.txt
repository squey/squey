#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

configure_file(test_issues.sh.cmake test_issues.sh @ONLY)

set(LINK_LIBRARIES pvkernel picviz ${QT_LIBRARIES} ${TBB_LIBRARIES} ${ICU_LIBRARY} ${OpenMP_LIBRARIES} ${OPENGL_LIBRARIES})

enable_testing()

declare_pv_test(Tpicviz_root root.cpp)
target_link_libraries(Tpicviz_root ${LINK_LIBRARIES})
add_test(TestRoot Tpicviz_root)

#declare_pv_test(Tpicviz_bench_mapping_time bench_mapping-time.cpp)
#target_link_libraries(Tpicviz_bench_mapping_time ${LINK_LIBRARIES})

# FIXME : This test is a bit slow but it test a lot of thing. We should split
# it in multiple tests
declare_pv_test(TPVSelection PVSelection.cpp)
target_link_libraries(TPVSelection ${LINK_LIBRARIES})
add_test(TestPVSelection TPVSelection)

# Mapping testing (had_issues/mapping)
# FIXME : This test can't work as it compare with references which doesn't
# exists.
#declare_pv_test(Tpicviz_process_file_mapping process_file_mapping.cpp)
#target_link_libraries(Tpicviz_process_file_mapping ${LINK_LIBRARIES})
#add_test(TestIssuesMapping test_issues.sh mapping)

# Plotting testing (had_issues/plotting)
# FIXME : The plotting folder doesn't exist
#declare_pv_test(Tpicviz_process_file_plotting process_file_plotting.cpp)
#target_link_libraries(Tpicviz_process_file_plotting ${LINK_LIBRARIES})
#add_test(TestIssuesPlotting test_issues.sh plotting)

# Source serialization
declare_pv_test(Tpicviz_source_serialize source-serialize.cpp)
target_link_libraries(Tpicviz_source_serialize ${LINK_LIBRARIES})

# Source serialization
declare_pv_test(Tpicviz_root_serialize root-serialize.cpp)
target_link_libraries(Tpicviz_root_serialize ${LINK_LIBRARIES})

# Object sizes
declare_pv_test(Tpicviz_sizes sizes.cpp)
target_link_libraries(Tpicviz_sizes ${LINK_LIBRARIES})

if(ENABLE_CORRELATION)
	# PVAD2GView
	#declare_pv_test(Tpicviz_ad2gview ad2gview.cpp)
	#target_link_libraries(Tpicviz_ad2gview ${LINK_LIBRARIES} ${TULIP_LIBRARY})
endif()

declare_pv_test(sel_red selection_red.cpp)
target_link_libraries(sel_red ${LINK_LIBRARIES} ${TULIP_LIBRARY})

# Sparse selection
declare_pv_test(Tpicviz_sparse_sel sparse_selection.cpp)
target_link_libraries(Tpicviz_sparse_sel ${LINK_LIBRARIES})

# Forward declaration
declare_pv_test(Tpicviz_forward_decl forward_decl.cpp)
target_link_libraries(Tpicviz_forward_decl ${LINK_LIBRARIES})

# Mapping plugin test
#declare_pv_test(Tpicviz_plugin_mapping plugin-mapping.cpp)
#target_link_libraries(Tpicviz_plugin_mapping ${LINK_LIBRARIES})


# IP mapping performance
#declare_pv_test(Tpicviz_ip_mapping ip-mapping.cpp)
#target_link_libraries(Tpicviz_ip_mapping ${LINK_LIBRARIES})

declare_pv_test(Tpicviz_cat_plotted cat_plotted_file.cpp)
target_link_libraries(Tpicviz_cat_plotted ${LINK_LIBRARIES})

declare_pv_test(Tpicviz_process picviz-process.cpp)
target_link_libraries(Tpicviz_process ${LINK_LIBRARIES})

# New Layer Dialog
declare_pv_test(Tqt_new_layer_dialog new_layer_dialog.cpp)
target_link_libraries(Tqt_new_layer_dialog ${LINK_LIBRARIES})

# Export selection
declare_pv_test(Tpicviz_export_selection export_selection.cpp)
target_link_libraries(Tpicviz_export_selection  ${LINK_LIBRARIES})
add_test(TestExportSelection Tpicviz_export_selection ${TESTS_FILES_DIR}/sources/proxy.log ${TESTS_FILES_DIR}/formats/proxy.log.format)

declare_lib_tests_target()
