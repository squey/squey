#
# @file
#

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt6Core_EXECUTABLE_COMPILE_FLAGS}")

set(LINK_LIBRARIES pvkernel pvsquey)
include_directories(${PVLICENSE_INCLUDE_DIR})

add_squey_compile_test(Tsquey_heat_line LINKS ${LINK_LIBRARIES} FILES heat_line.cpp)
add_squey_test(Tsquey_heat_line Tsquey_heat_line)
add_bench(Tsquey_heat_line LINKS ${LINK_LIBRARIES} FILES heat_line.cpp)

add_squey_compile_test(Tsquey_multiple_search LINKS ${LINK_LIBRARIES} FILES multiple_search.cpp)
add_squey_test(Tsquey_multiple_search Tsquey_multiple_search)
add_bench(Tsquey_multiple_search LINKS ${LINK_LIBRARIES} FILES multiple_search.cpp)

add_squey_compile_test(Tsquey_errors_search LINKS ${LINK_LIBRARIES} FILES errors_search.cpp)
add_squey_test(Tsquey_errors_search Tsquey_errors_search)
add_bench(Tsquey_errors_search LINKS ${LINK_LIBRARIES} FILES errors_search.cpp)

add_squey_compile_test(Tsquey_format LINKS ${LINK_LIBRARIES} FILES format.cpp)
add_squey_test(Tsquey_format Tsquey_format)

add_squey_compile_test(Tqt_new_layer_dialog LINKS ${LINK_LIBRARIES} FILES new_layer_dialog.cpp)

add_squey_compile_test(Tsquey_export_selection LINKS ${LINK_LIBRARIES} FILES export_selection.cpp)
add_squey_test(Tsquey_export_selection Tsquey_export_selection ${TESTS_FILES_DIR}/sources/proxy.log ${TESTS_FILES_DIR}/formats/proxy.log.format 0 ${TESTS_FILES_DIR}/sources/proxy.log)

add_squey_test(Tsquey_export_small_selection_gz Tsquey_export_selection ${TESTS_FILES_DIR}/sources/proxy.log.gz ${TESTS_FILES_DIR}/formats/proxy_small.log.format 0 ${TESTS_FILES_DIR}/sources/proxy_small.log)
add_squey_test(Tsquey_export_small_selection_bz2 Tsquey_export_selection ${TESTS_FILES_DIR}/sources/proxy.log.bz2 ${TESTS_FILES_DIR}/formats/proxy_small.log.format 0 ${TESTS_FILES_DIR}/sources/proxy_small.log)
add_squey_test(Tsquey_export_small_selection_zip Tsquey_export_selection ${TESTS_FILES_DIR}/sources/proxy.log.zip ${TESTS_FILES_DIR}/formats/proxy_small.log.format 0 ${TESTS_FILES_DIR}/sources/proxy_small.log)
add_squey_test(Tsquey_export_small_selection_xz Tsquey_export_selection ${TESTS_FILES_DIR}/sources/proxy.log.xz ${TESTS_FILES_DIR}/formats/proxy_small.log.format 0 ${TESTS_FILES_DIR}/sources/proxy_small.log)
add_squey_test(Tsquey_export_small_selection_zst Tsquey_export_selection ${TESTS_FILES_DIR}/sources/proxy.log.zst ${TESTS_FILES_DIR}/formats/proxy_small.log.format 0 ${TESTS_FILES_DIR}/sources/proxy_small.log)


# Test selection
add_squey_test(Tsquey_export_odd_selection Tsquey_export_selection ${TESTS_FILES_DIR}/sources/proxy.log ${TESTS_FILES_DIR}/formats/proxy.log.format 1 ${TESTS_FILES_DIR}/sources/proxy_odd.log)

add_squey_test(Tsquey_formatter_number Tsquey_export_selection ${TESTS_FILES_DIR}/picviz/formatter_number.csv ${TESTS_FILES_DIR}/picviz/formatter_number.csv.format 0)

add_squey_compile_test(Tmapping_float_default LINKS ${LINK_LIBRARIES} FILES mapping_copy.cpp)
add_squey_test(Tmapping_float_default Tmapping_float_default ${TESTS_FILES_DIR}/picviz/float_default_mapping.csv ${TESTS_FILES_DIR}/picviz/float_default_mapping.csv.format)
add_bench(Tmapping_float_default LINKS ${LINK_LIBRARIES} FILES mapping_copy.cpp)

add_squey_compile_test(Tmapping_double_default LINKS ${LINK_LIBRARIES} FILES mapping_copy.cpp)
add_squey_test(Tmapping_double_default Tmapping_double_default ${TESTS_FILES_DIR}/picviz/double_default_mapping.csv ${TESTS_FILES_DIR}/picviz/double_default_mapping.csv.format)
add_bench(Tmapping_double_default LINKS ${LINK_LIBRARIES} FILES mapping_copy.cpp)

add_squey_compile_test(Tmapping_host_default LINKS ${LINK_LIBRARIES} FILES mapping_host_default.cpp)
add_squey_test(Tmapping_host_default Tmapping_host_default)
add_bench(Tmapping_host_default LINKS ${LINK_LIBRARIES} FILES mapping_host_default.cpp)

add_squey_compile_test(Tmapping_integer_default LINKS ${LINK_LIBRARIES} FILES mapping_copy.cpp)
add_squey_test(Tmapping_integer_default Tmapping_integer_default ${TESTS_FILES_DIR}/picviz/integer_default_mapping.csv ${TESTS_FILES_DIR}/picviz/integer_default_mapping.csv.format)
add_bench(Tmapping_integer_default LINKS ${LINK_LIBRARIES} FILES mapping_copy.cpp)

add_squey_compile_test(Tmapping_integer_unsigned LINKS ${LINK_LIBRARIES} FILES mapping_copy.cpp)
add_squey_test(Tmapping_integer_unsigned Tmapping_integer_unsigned ${TESTS_FILES_DIR}/picviz/integer_unsigned_mapping.csv ${TESTS_FILES_DIR}/picviz/integer_unsigned_mapping.csv.format)
add_bench(Tmapping_integer_unsigned LINKS ${LINK_LIBRARIES} FILES mapping_copy.cpp)

add_squey_compile_test(Tmapping_ipv4_default LINKS ${LINK_LIBRARIES} FILES mapping_copy.cpp)
add_squey_test(Tmapping_ipv4_default Tmapping_ipv4_default ${TESTS_FILES_DIR}/picviz/ipv4_default_mapping.csv ${TESTS_FILES_DIR}/picviz/ipv4_default_mapping.csv.format)
add_bench(Tmapping_ipv4_default LINKS ${LINK_LIBRARIES} FILES mapping_copy.cpp)

add_squey_compile_test(Tscaling_ipv4_uniform LINKS ${LINK_LIBRARIES} FILES scaling_uniform.cpp)
add_squey_test(Tscaling_ipv4_uniform Tscaling_ipv4_uniform ${TESTS_FILES_DIR}/picviz/ipv4_default_mapping.csv ${TESTS_FILES_DIR}/picviz/ipv4_default_mapping.csv.format)
add_bench(Tscaling_ipv4_uniform LINKS ${LINK_LIBRARIES} FILES scaling_uniform.cpp)

add_squey_compile_test(Tscaling_ipv6_uniform LINKS ${LINK_LIBRARIES} FILES scaling_uniform.cpp)
add_squey_test(Tscaling_ipv6_uniform Tscaling_ipv6_uniform ${TESTS_FILES_DIR}/picviz/ipv6_10000.csv ${TESTS_FILES_DIR}/picviz/ipv6.csv.format)
add_bench(Tscaling_ipv6_uniform LINKS ${LINK_LIBRARIES} FILES scaling_uniform.cpp)

add_squey_compile_test(Tmapping_string_4bsort LINKS ${LINK_LIBRARIES} FILES mapping_string_4bsort.cpp)
add_squey_test(Tmapping_string_4bsort Tmapping_string_4bsort)
add_bench(Tmapping_string_4bsort LINKS ${LINK_LIBRARIES} FILES mapping_string_4bsort.cpp)

add_squey_compile_test(Tmapping_string_default LINKS ${LINK_LIBRARIES} FILES mapping_string_default.cpp)
add_squey_test(Tmapping_string_default Tmapping_string_default)
add_bench(Tmapping_string_default LINKS ${LINK_LIBRARIES} FILES mapping_string_default.cpp)

add_squey_compile_test(Tmapping_time_default LINKS ${LINK_LIBRARIES} FILES mapping_time_default.cpp)
add_squey_test(Tmapping_time_default Tmapping_time_default)
add_bench(Tmapping_time_default LINKS ${LINK_LIBRARIES} FILES mapping_time_default.cpp)

add_squey_compile_test(Tmapping_time_ms_default LINKS ${LINK_LIBRARIES} FILES mapping_time_ms_default.cpp)
add_squey_test(Tmapping_time_ms_default Tmapping_time_ms_default)
add_bench(Tmapping_time_ms_default LINKS ${LINK_LIBRARIES} FILES mapping_time_ms_default.cpp)

add_squey_compile_test(Tmapping_time_us_default LINKS ${LINK_LIBRARIES} FILES mapping_time_us_default.cpp)
add_squey_test(Tmapping_time_us_default Tmapping_time_us_default)
add_bench(Tmapping_time_us_default LINKS ${LINK_LIBRARIES} FILES mapping_time_us_default.cpp)

add_squey_compile_test(Tmapping_time_24h LINKS ${LINK_LIBRARIES} FILES mapping_time_24h.cpp)
add_squey_test(Tmapping_time_24h Tmapping_time_24h)
add_bench(Tmapping_time_24h LINKS ${LINK_LIBRARIES} FILES mapping_time_24h.cpp)

add_squey_compile_test(Tmapping_time_ms_24h LINKS ${LINK_LIBRARIES} FILES mapping_time_ms_24h.cpp)
add_squey_test(Tmapping_time_ms_24h Tmapping_time_ms_24h)
add_bench(Tmapping_time_ms_24h LINKS ${LINK_LIBRARIES} FILES mapping_time_ms_24h.cpp)

add_squey_compile_test(Tmapping_time_us_24h LINKS ${LINK_LIBRARIES} FILES mapping_time_us_24h.cpp)
add_squey_test(Tmapping_time_us_24h Tmapping_time_us_24h)
add_bench(Tmapping_time_us_24h LINKS ${LINK_LIBRARIES} FILES mapping_time_us_24h.cpp)

add_squey_compile_test(Tmapping_time_week LINKS ${LINK_LIBRARIES} FILES mapping_time_week.cpp)
add_squey_test(Tmapping_time_week Tmapping_time_week)
add_bench(Tmapping_time_week LINKS ${LINK_LIBRARIES} FILES mapping_time_week.cpp)

add_squey_compile_test(Tmapping_time_ms_week LINKS ${LINK_LIBRARIES} FILES mapping_time_ms_week.cpp)
add_squey_test(Tmapping_time_ms_week Tmapping_time_ms_week)
add_bench(Tmapping_time_ms_week LINKS ${LINK_LIBRARIES} FILES mapping_time_ms_week.cpp)

add_squey_compile_test(Tmapping_time_us_week LINKS ${LINK_LIBRARIES} FILES mapping_time_us_week.cpp)
add_squey_test(Tmapping_time_us_week Tmapping_time_us_week)
add_bench(Tmapping_time_us_week LINKS ${LINK_LIBRARIES} FILES mapping_time_us_week.cpp)

add_squey_compile_test(Tmapping_mac_address LINKS ${LINK_LIBRARIES} FILES mapping_mac_address.cpp)
add_squey_test(Tmapping_mac_address_lin Tmapping_mac_address 2 ${TESTS_FILES_DIR}/picviz/mac_address_lin_mapping.format)
add_bench(Tmapping_mac_address_lin LINKS ${LINK_LIBRARIES} FILES mapping_mac_address.cpp)
add_squey_test(Tmapping_mac_address_lin-uni Tmapping_mac_address 4 ${TESTS_FILES_DIR}/picviz/mac_address_lin-uni_mapping.format)
add_bench(Tmapping_mac_address_lin LINKS ${LINK_LIBRARIES} FILES mapping_mac_address.cpp)
add_squey_test(Tmapping_mac_address_uni-lin Tmapping_mac_address 5 ${TESTS_FILES_DIR}/picviz/mac_address_uni-lin_mapping.format)
add_bench(Tmapping_mac_address_uni-lin LINKS ${LINK_LIBRARIES} FILES mapping_mac_address.cpp)
add_squey_test(Tmapping_mac_address_uni-uni Tmapping_mac_address 6 ${TESTS_FILES_DIR}/picviz/mac_address_uni-uni_mapping.format)
add_bench(Tmapping_mac_address_uni-uni LINKS ${LINK_LIBRARIES} FILES mapping_mac_address.cpp)

add_squey_compile_test(Tsquey_investigation LINKS ${LINK_LIBRARIES} FILES investigation.cpp)
add_squey_test(Tsquey_investigation Tsquey_investigation)

add_squey_compile_test(Tsquey_investigation_multisource LINKS ${LINK_LIBRARIES} FILES investigation_multisource.cpp)
add_squey_test(Tsquey_investigation_multisource Tsquey_investigation_multisource)
set_tests_properties(SQUEY_TEST_Tsquey_investigation_multisource PROPERTIES DEPENDS SQUEY_TEST_Tsquey_investigation)

add_squey_compile_test(Tscaling_integer_default_default LINKS ${LINK_LIBRARIES} FILES scaling.cpp)
add_squey_test(Tscaling_integer_default_default Tscaling_integer_default_default ${TESTS_FILES_DIR}/picviz/integer_default_mapping.csv ${TESTS_FILES_DIR}/picviz/integer_default_mapping.csv.format)
add_bench(Tscaling_integer_default_default LINKS ${LINK_LIBRARIES} FILES scaling.cpp)

add_squey_compile_test(Tscaling_integer_unsigned_default LINKS ${LINK_LIBRARIES} FILES scaling.cpp)
add_squey_test(Tscaling_integer_unsigned_default Tscaling_integer_unsigned_default ${TESTS_FILES_DIR}/picviz/integer_unsigned_mapping.csv ${TESTS_FILES_DIR}/picviz/integer_unsigned_mapping.csv.format)
add_bench(Tscaling_integer_unsigned_default LINKS ${LINK_LIBRARIES} FILES scaling.cpp)

add_squey_compile_test(Tscaling_float_default_default LINKS ${LINK_LIBRARIES} FILES scaling.cpp)
add_squey_test(Tscaling_float_default_default Tscaling_float_default_default ${TESTS_FILES_DIR}/picviz/float_default_mapping.csv ${TESTS_FILES_DIR}/picviz/float_default_mapping.csv.format)
add_bench(Tscaling_float_default_default LINKS ${LINK_LIBRARIES} FILES scaling.cpp)

add_squey_compile_test(Tscaling_double_default_default LINKS ${LINK_LIBRARIES} FILES scaling.cpp)
add_squey_test(Tscaling_double_default_default Tscaling_double_default_default ${TESTS_FILES_DIR}/picviz/double_default_mapping.csv ${TESTS_FILES_DIR}/picviz/double_default_mapping.csv.format)
add_bench(Tscaling_double_default_default LINKS ${LINK_LIBRARIES} FILES scaling.cpp)

add_squey_compile_test(Tscaling_string_default_enum LINKS ${LINK_LIBRARIES} FILES scaling_string_default_enum.cpp)
add_squey_test(Tscaling_string_default_enum Tscaling_string_default_enum)
add_bench(Tscaling_string_default_enum LINKS ${LINK_LIBRARIES} FILES scaling_string_default_enum.cpp)

add_squey_compile_test(Tscaling_time_24h_minmax LINKS ${LINK_LIBRARIES} FILES scaling_time_24h_minmax.cpp)
add_squey_test(Tscaling_time_24h_minmax Tscaling_time_24h_minmax)
add_bench(Tscaling_time_24h_minmax LINKS ${LINK_LIBRARIES} FILES scaling_time_24h_minmax.cpp)

add_squey_compile_test(Tscaling_time_week_minmax LINKS ${LINK_LIBRARIES} FILES scaling_time_week_minmax.cpp)
add_squey_test(Tscaling_time_week_minmax Tscaling_time_week_minmax)
add_bench(Tscaling_time_week_minmax LINKS ${LINK_LIBRARIES} FILES scaling_time_week_minmax.cpp)

add_squey_compile_test(Tscaling_integer_default_log LINKS ${LINK_LIBRARIES} FILES scaling.cpp)
add_squey_test(Tscaling_integer_default_log Tscaling_integer_default_log ${TESTS_FILES_DIR}/picviz/integer_default_mapping.csv ${TESTS_FILES_DIR}/picviz/integer_default_mapping.csv.format)
add_bench(Tscaling_integer_default_log LINKS ${LINK_LIBRARIES} FILES scaling.cpp)

add_squey_compile_test(Tscaling_integer_unsigned_log LINKS ${LINK_LIBRARIES} FILES scaling.cpp)
add_squey_test(Tscaling_integer_unsigned_log Tscaling_integer_unsigned_log ${TESTS_FILES_DIR}/picviz/integer_unsigned_mapping.csv ${TESTS_FILES_DIR}/picviz/integer_unsigned_mapping.csv.format)
add_bench(Tscaling_integer_unsigned_log LINKS ${LINK_LIBRARIES} FILES scaling.cpp)

add_squey_compile_test(Tscaling_float_default_log LINKS ${LINK_LIBRARIES} FILES scaling.cpp)
add_squey_test(Tscaling_float_default_log Tscaling_float_default_log ${TESTS_FILES_DIR}/picviz/float_default_mapping.csv ${TESTS_FILES_DIR}/picviz/float_default_mapping.csv.format)
add_bench(Tscaling_float_default_log LINKS ${LINK_LIBRARIES} FILES scaling.cpp)

add_squey_compile_test(Tscaling_time_24h_log LINKS ${LINK_LIBRARIES} FILES scaling_time_24h_log.cpp)
add_squey_test(Tscaling_time_24h_log Tscaling_time_24h_log)
add_bench(Tscaling_time_24h_log LINKS ${LINK_LIBRARIES} FILES scaling_time_24h_log.cpp)

add_squey_compile_test(Tscaling_time_week_log LINKS ${LINK_LIBRARIES} FILES scaling_time_week_log.cpp)
add_squey_test(Tscaling_time_week_log Tscaling_time_week_log)
add_bench(Tscaling_time_week_log LINKS ${LINK_LIBRARIES} FILES scaling_time_week_log.cpp)

add_squey_compile_test(Tscaling_port LINKS ${LINK_LIBRARIES} FILES scaling_port.cpp)
add_squey_test(Tscaling_port Tscaling_port)
add_bench(Tscaling_port LINKS ${LINK_LIBRARIES} FILES scaling_port.cpp)

add_squey_compile_test(Tscaling_integer_default_enum LINKS ${LINK_LIBRARIES} FILES scaling.cpp)
add_squey_test(Tscaling_integer_default_enum Tscaling_integer_default_enum ${TESTS_FILES_DIR}/picviz/integer_default_mapping.csv ${TESTS_FILES_DIR}/picviz/integer_default_enum.csv.format)
add_bench(Tscaling_integer_default_enum LINKS ${LINK_LIBRARIES} FILES scaling.cpp)

# FIXME : This test is a bit slow but it test a lot of thing. We should split
# it in multiple tests
add_squey_compile_test(TPVSelection LINKS ${LINK_LIBRARIES} FILES PVSelection.cpp)
add_squey_test(TPVSelection TPVSelection)

add_squey_compile_test(Tsquey_bad_convertions_merge LINKS ${LINK_LIBRARIES} FILES bad_convertions_merge.cpp)
add_squey_test(Tsquey_bad_convertions_merge Tsquey_bad_convertions_merge)

add_squey_compile_test(Tsquey_correlation LINKS ${LINK_LIBRARIES} FILES correlation.cpp)
add_squey_test(Tsquey_correlation Tsquey_correlation)
add_bench(Tsquey_correlation LINKS ${LINK_LIBRARIES} FILES correlation.cpp)

add_squey_compile_test(Trush_multi_chunk LINKS ${LINK_LIBRARIES} FILES multi_chunk.cpp)
add_squey_test(Trush_multi_chunk Trush_multi_chunk)

add_squey_compile_test(Trush_multi_chunk_header LINKS ${LINK_LIBRARIES} FILES multi_chunk_header.cpp)
add_squey_test(Trush_multi_chunk_header Trush_multi_chunk_header)

add_squey_compile_test(Tsquey_export_scaled LINKS ${LINK_LIBRARIES} FILES export_scaled.cpp)
add_squey_test(Tsquey_export_scaled Tsquey_export_scaled ${TESTS_FILES_DIR}/picviz/export_scaled.csv.in ${TESTS_FILES_DIR}/picviz/export_scaled.csv.out ${TESTS_FILES_DIR}/picviz/export_scaled.csv.format)
add_bench(Tsquey_export_scaled LINKS ${LINK_LIBRARIES} FILES export_scaled.cpp)

add_squey_compile_test(Tsquey_scaling_one_value LINKS ${LINK_LIBRARIES} FILES scaling_one_value.cpp)
add_squey_test(Tsquey_scaling_one_value Tsquey_scaling_one_value ${TESTS_FILES_DIR}/picviz/scaling_one_value.csv ${TESTS_FILES_DIR}/picviz/scaling_one_value.csv.format)

add_squey_compile_test(Tsquey_axis_gradient LINKS ${LINK_LIBRARIES} FILES axis_gradient.cpp)
add_squey_test(Tsquey_axis_gradient Tsquey_axis_gradient ${TESTS_FILES_DIR}/picviz/axis_gradient.csv ${TESTS_FILES_DIR}/picviz/axis_gradient.csv.format)
add_bench(Tsquey_axis_gradient LINKS ${LINK_LIBRARIES} FILES axis_gradient.cpp)

add_squey_compile_test(Tsquey_axes_combination LINKS ${LINK_LIBRARIES} FILES axes_combination.cpp)
add_squey_test(Tsquey_axes_combination Tsquey_axes_combination)

add_squey_compile_test(Tsquey_view_unselected_zombie_toggle LINKS ${LINK_LIBRARIES} FILES view_unselected_zombie_toggle.cpp)
add_squey_test(Tsquey_view_unselected_zombie_toggle Tsquey_view_unselected_zombie_toggle)

# Cancel import
add_squey_compile_test(Tsquey_cancel_import LINKS ${LINK_LIBRARIES} FILES cancel_import.cpp)
add_squey_test(Tsquey_cancel_import Tsquey_cancel_import ${TESTS_FILES_DIR}/sources/proxy.log.gz ${TESTS_FILES_DIR}/formats/proxy.log.format)

# Cancel export
add_squey_compile_test(Tsquey_cancel_export LINKS ${LINK_LIBRARIES} FILES cancel_export.cpp)
add_squey_test(Tsquey_cancel_export Tsquey_cancel_export ${TESTS_FILES_DIR}/sources/proxy.log.gz ${TESTS_FILES_DIR}/formats/proxy.log.format)

# Range subsampler
add_squey_compile_test(Tsquey_range_subsampler LINKS ${LINK_LIBRARIES} FILES range_subsampler.cpp)
add_squey_test(Tsquey_range_subsampler Tsquey_range_subsampler)

# multi-inputs
add_squey_compile_test(Tsquey_multi_inputs LINKS ${LINK_LIBRARIES} FILES multi_inputs.cpp)
add_squey_test(Tsquey_multi_inputs Tsquey_multi_inputs)

# python_scripting
add_squey_compile_test(Tsquey_python_scripting LINKS ${LINK_LIBRARIES} FILES python_scripting.cpp)
add_squey_test(Tsquey_python_scripting Tsquey_python_scripting ${TESTS_FILES_DIR}/sources/proxy_sample1.log ${TESTS_FILES_DIR}/sources/proxy_sample1_with_python_script.log.format ${TESTS_FILES_DIR}/exports/Tsquey_python_scripting)
add_squey_test(Tsquey_embeded_python_scripting Tsquey_python_scripting ${TESTS_FILES_DIR}/sources/proxy_sample1.log ${TESTS_FILES_DIR}/sources/proxy_sample1_with_embeded_python_script.log.format ${TESTS_FILES_DIR}/exports/Tsquey_python_scripting)

file(RELATIVE_PATH REL_TEST_DIR ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake DESTINATION "tests/${REL_TEST_DIR}" COMPONENT tests)