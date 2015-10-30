#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

configure_file(test_issues.sh.cmake test_issues.sh @ONLY)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Core_EXECUTABLE_COMPILE_FLAGS}")

set(LINK_LIBRARIES pvkernel picviz ${TBB_LIBRARIES} ${ICU_LIBRARY}
				   ${OpenMP_LIBRARIES} ${OPENGL_LIBRARIES})

# FIXME : This test can't work as it compare with references which doesn't
# exists.
add_pv_compile_test(Tpicviz_process_file_mapping LINKS ${LINK_LIBRARIES} FILES process_file_mapping.cpp)

# FIXME : The plotting folder doesn't exist
add_pv_compile_test(Tpicviz_process_file_plotting LINKS ${LINK_LIBRARIES} FILES process_file_plotting.cpp)

add_pv_compile_test(Tpicviz_source_serialize LINKS ${LINK_LIBRARIES} FILES source-serialize.cpp)
add_pv_compile_test(Tpicviz_root_serialize LINKS ${LINK_LIBRARIES} FILES root-serialize.cpp)
add_pv_compile_test(Tpicviz_sizes LINKS ${LINK_LIBRARIES} FILES sizes.cpp)
add_pv_compile_test(sel_red LINKS ${LINK_LIBRARIES} FILES selection_red.cpp)
add_pv_compile_test(Tpicviz_sparse_sel LINKS ${LINK_LIBRARIES} FILES sparse_selection.cpp)
add_pv_compile_test(Tpicviz_forward_decl LINKS ${LINK_LIBRARIES} FILES forward_decl.cpp)
add_pv_compile_test(Tpicviz_cat_plotted LINKS ${LINK_LIBRARIES} FILES cat_plotted_file.cpp)
add_pv_compile_test(Tpicviz_process LINKS ${LINK_LIBRARIES} FILES picviz-process.cpp)
add_pv_compile_test(Tqt_new_layer_dialog LINKS ${LINK_LIBRARIES} FILES new_layer_dialog.cpp)
add_pv_compile_test(Tpicviz_export_selection LINKS ${LINK_LIBRARIES} FILES export_selection.cpp)
add_test(Tpicviz_export_selection Tpicviz_export_selection ${TESTS_FILES_DIR}/sources/proxy.log ${TESTS_FILES_DIR}/formats/proxy.log.format)

add_pv_compile_test(Tpicviz_root LINKS ${LINK_LIBRARIES} FILES root.cpp)
add_test(Tpicviz_root Tpicviz_root)

# FIXME : This test is a bit slow but it test a lot of thing. We should split
# it in multiple tests
add_pv_compile_test(TPVSelection LINKS ${LINK_LIBRARIES} FILES PVSelection.cpp)
add_test(TPVSelection TPVSelection)
