#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

configure_file(test_issues.sh.cmake test_issues.sh @ONLY)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Core_EXECUTABLE_COMPILE_FLAGS}")

set(LINK_LIBRARIES pvkernel inendi ${TBB_LIBRARIES} ${ICU_LIBRARY}
				   ${OpenMP_LIBRARIES} ${OPENGL_LIBRARIES})

# FIXME : This test can't work as it compare with references which doesn't
# exists.
add_pv_compile_test(Tinendi_process_file_mapping LINKS ${LINK_LIBRARIES} FILES process_file_mapping.cpp)

# FIXME : The plotting folder doesn't exist
add_pv_compile_test(Tinendi_process_file_plotting LINKS ${LINK_LIBRARIES} FILES process_file_plotting.cpp)

add_pv_compile_test(Tinendi_heat_line LINKS ${LINK_LIBRARIES} FILES heat_line.cpp)
add_test(Tinendi_heat_line Tinendi_heat_line)
add_bench(Tinendi_heat_line LINKS ${LINK_LIBRARIES} FILES heat_line.cpp)

add_pv_compile_test(Tinendi_multiple_search LINKS ${LINK_LIBRARIES} FILES multiple_search.cpp)
add_test(Tinendi_multiple_search Tinendi_multiple_search)
add_bench(Tinendi_multiple_search LINKS ${LINK_LIBRARIES} FILES multiple_search.cpp)

add_pv_compile_test(Tinendi_format LINKS ${LINK_LIBRARIES} FILES format.cpp)
add_test(Tinendi_format Tinendi_format)

add_pv_compile_test(Tinendi_source_serialize LINKS ${LINK_LIBRARIES} FILES source-serialize.cpp)
add_pv_compile_test(Tinendi_root_serialize LINKS ${LINK_LIBRARIES} FILES root-serialize.cpp)
add_pv_compile_test(sel_red LINKS ${LINK_LIBRARIES} FILES selection_red.cpp)
add_pv_compile_test(Tinendi_forward_decl LINKS ${LINK_LIBRARIES} FILES forward_decl.cpp)
add_pv_compile_test(Tinendi_cat_plotted LINKS ${LINK_LIBRARIES} FILES cat_plotted_file.cpp)
add_pv_compile_test(Tinendi_process LINKS ${LINK_LIBRARIES} FILES inendi-process.cpp)
add_pv_compile_test(Tqt_new_layer_dialog LINKS ${LINK_LIBRARIES} FILES new_layer_dialog.cpp)
add_pv_compile_test(Tinendi_export_selection LINKS ${LINK_LIBRARIES} FILES export_selection.cpp)
add_test(Tinendi_export_selection Tinendi_export_selection ${TESTS_FILES_DIR}/sources/proxy.log ${TESTS_FILES_DIR}/formats/proxy.log.format)
if(MINESET)
	add_pv_compile_test(Tinendi_export_selection_to_mineset LINKS ${LINK_LIBRARIES} FILES export_selection_to_mineset.cpp)
	add_test(Tinendi_export_selection_to_mineset Tinendi_export_selection_to_mineset ${TESTS_FILES_DIR}/sources/proxy_mineset.log ${TESTS_FILES_DIR}/formats/proxy.log.format)
endif(MINESET)
add_test(Tinendi_formatter_number Tinendi_export_selection ${TESTS_FILES_DIR}/picviz/formatter_number.csv ${TESTS_FILES_DIR}/picviz/formatter_number.csv.format)

add_pv_compile_test(Tmapping_enum_default LINKS ${LINK_LIBRARIES} FILES mapping_enum_default.cpp)
add_test(Tmapping_enum_default Tmapping_enum_default)
add_bench(Tmapping_enum_default LINKS ${LINK_LIBRARIES} FILES mapping_enum_default.cpp)

add_pv_compile_test(Tmapping_float_default LINKS ${LINK_LIBRARIES} FILES mapping_float_default.cpp)
add_test(Tmapping_float_default Tmapping_float_default)
add_bench(Tmapping_float_default LINKS ${LINK_LIBRARIES} FILES mapping_float_default.cpp)

add_pv_compile_test(Tmapping_host_default LINKS ${LINK_LIBRARIES} FILES mapping_host_default.cpp)
add_test(Tmapping_host_default Tmapping_host_default)
add_bench(Tmapping_host_default LINKS ${LINK_LIBRARIES} FILES mapping_host_default.cpp)

add_pv_compile_test(Tmapping_integer_default LINKS ${LINK_LIBRARIES} FILES mapping_integer_default.cpp)
add_test(Tmapping_integer_default Tmapping_integer_default)
add_bench(Tmapping_integer_default LINKS ${LINK_LIBRARIES} FILES mapping_integer_default.cpp)

add_pv_compile_test(Tmapping_integer_unsigned LINKS ${LINK_LIBRARIES} FILES mapping_integer_unsigned.cpp)
add_test(Tmapping_integer_unsigned Tmapping_integer_unsigned)
add_bench(Tmapping_integer_unsigned LINKS ${LINK_LIBRARIES} FILES mapping_integer_unsigned.cpp)

add_pv_compile_test(Tmapping_ipv4_default LINKS ${LINK_LIBRARIES} FILES mapping_ipv4_default.cpp)
add_test(Tmapping_ipv4_default Tmapping_ipv4_default)
add_bench(Tmapping_ipv4_default LINKS ${LINK_LIBRARIES} FILES mapping_ipv4_default.cpp)

add_pv_compile_test(Tmapping_ipv4_uniform LINKS ${LINK_LIBRARIES} FILES mapping_ipv4_uniform.cpp)
add_test(Tmapping_ipv4_uniform Tmapping_ipv4_uniform)
add_bench(Tmapping_ipv4_uniform LINKS ${LINK_LIBRARIES} FILES mapping_ipv4_uniform.cpp)

add_pv_compile_test(Tmapping_string_4bsort LINKS ${LINK_LIBRARIES} FILES mapping_string_4bsort.cpp)
add_test(Tmapping_string_4bsort Tmapping_string_4bsort)
add_bench(Tmapping_string_4bsort LINKS ${LINK_LIBRARIES} FILES mapping_string_4bsort.cpp)

add_pv_compile_test(Tmapping_string_default LINKS ${LINK_LIBRARIES} FILES mapping_string_default.cpp)
add_test(Tmapping_string_default Tmapping_string_default)
add_bench(Tmapping_string_default LINKS ${LINK_LIBRARIES} FILES mapping_string_default.cpp)

add_pv_compile_test(Tmapping_time_default LINKS ${LINK_LIBRARIES} FILES mapping_time_default.cpp)
add_test(Tmapping_time_default Tmapping_time_default)
add_bench(Tmapping_time_default LINKS ${LINK_LIBRARIES} FILES mapping_time_default.cpp)

add_pv_compile_test(Tmapping_time_ms_default LINKS ${LINK_LIBRARIES} FILES mapping_time_ms_default.cpp)
add_test(Tmapping_time_ms_default Tmapping_time_ms_default)
add_bench(Tmapping_time_ms_default LINKS ${LINK_LIBRARIES} FILES mapping_time_ms_default.cpp)

add_pv_compile_test(Tmapping_time_us_default LINKS ${LINK_LIBRARIES} FILES mapping_time_us_default.cpp)
add_test(Tmapping_time_us_default Tmapping_time_us_default)
add_bench(Tmapping_time_us_default LINKS ${LINK_LIBRARIES} FILES mapping_time_us_default.cpp)

add_pv_compile_test(Tmapping_time_24h LINKS ${LINK_LIBRARIES} FILES mapping_time_24h.cpp)
add_test(Tmapping_time_24h Tmapping_time_24h)
add_bench(Tmapping_time_24h LINKS ${LINK_LIBRARIES} FILES mapping_time_24h.cpp)

add_pv_compile_test(Tmapping_time_ms_24h LINKS ${LINK_LIBRARIES} FILES mapping_time_ms_24h.cpp)
add_test(Tmapping_time_ms_24h Tmapping_time_ms_24h)
add_bench(Tmapping_time_ms_24h LINKS ${LINK_LIBRARIES} FILES mapping_time_ms_24h.cpp)

add_pv_compile_test(Tmapping_time_us_24h LINKS ${LINK_LIBRARIES} FILES mapping_time_us_24h.cpp)
add_test(Tmapping_time_us_24h Tmapping_time_us_24h)
add_bench(Tmapping_time_us_24h LINKS ${LINK_LIBRARIES} FILES mapping_time_us_24h.cpp)

add_pv_compile_test(Tmapping_time_week LINKS ${LINK_LIBRARIES} FILES mapping_time_week.cpp)
add_test(Tmapping_time_week Tmapping_time_week)
add_bench(Tmapping_time_week LINKS ${LINK_LIBRARIES} FILES mapping_time_week.cpp)

add_pv_compile_test(Tmapping_time_ms_week LINKS ${LINK_LIBRARIES} FILES mapping_time_ms_week.cpp)
add_test(Tmapping_time_ms_week Tmapping_time_ms_week)
add_bench(Tmapping_time_ms_week LINKS ${LINK_LIBRARIES} FILES mapping_time_ms_week.cpp)

add_pv_compile_test(Tmapping_time_us_week LINKS ${LINK_LIBRARIES} FILES mapping_time_us_week.cpp)
add_test(Tmapping_time_us_week Tmapping_time_us_week)
add_bench(Tmapping_time_us_week LINKS ${LINK_LIBRARIES} FILES mapping_time_us_week.cpp)

add_pv_compile_test(Tinendi_investigation LINKS ${LINK_LIBRARIES} FILES investigation.cpp)
add_test(Tinendi_investigation Tinendi_investigation)

# FIXME : This test is a bit slow but it test a lot of thing. We should split
# it in multiple tests
add_pv_compile_test(TPVSelection LINKS ${LINK_LIBRARIES} FILES PVSelection.cpp)
add_test(TPVSelection TPVSelection)

add_pv_compile_test(Tinendi_bad_convertions_merge LINKS ${LINK_LIBRARIES} FILES bad_convertions_merge.cpp)
add_test(Tinendi_bad_convertions_merge Tinendi_bad_convertions_merge)

add_pv_compile_test(Tinendi_correlation LINKS ${LINK_LIBRARIES} FILES correlation.cpp)
add_test(Tinendi_correlation Tinendi_correlation)
add_bench(Tinendi_correlation LINKS ${LINK_LIBRARIES} FILES correlation.cpp)
