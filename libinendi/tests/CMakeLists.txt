#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Core_EXECUTABLE_COMPILE_FLAGS}")

set(LINK_LIBRARIES pvkernel ${PVLICENSE_LIBRARIES} inendi)
include_directories(${PVLICENSE_INCLUDE_DIR})

add_inspector_compile_test(Tinendi_heat_line LINKS ${LINK_LIBRARIES} FILES heat_line.cpp)
add_inspector_test(Tinendi_heat_line Tinendi_heat_line)
add_bench(Tinendi_heat_line LINKS ${LINK_LIBRARIES} FILES heat_line.cpp)

add_inspector_compile_test(Tinendi_multiple_search LINKS ${LINK_LIBRARIES} FILES multiple_search.cpp)
add_inspector_test(Tinendi_multiple_search Tinendi_multiple_search)
add_bench(Tinendi_multiple_search LINKS ${LINK_LIBRARIES} FILES multiple_search.cpp)

add_inspector_compile_test(Tinendi_errors_search LINKS ${LINK_LIBRARIES} FILES errors_search.cpp)
add_inspector_test(Tinendi_errors_search Tinendi_errors_search)
add_bench(Tinendi_errors_search LINKS ${LINK_LIBRARIES} FILES errors_search.cpp)

add_inspector_compile_test(Tinendi_format LINKS ${LINK_LIBRARIES} FILES format.cpp)
add_inspector_test(Tinendi_format Tinendi_format)

add_inspector_compile_test(Tqt_new_layer_dialog LINKS ${LINK_LIBRARIES} FILES new_layer_dialog.cpp)

add_inspector_compile_test(Tinendi_export_selection LINKS ${LINK_LIBRARIES} FILES export_selection.cpp)
add_inspector_test(Tinendi_export_selection Tinendi_export_selection ${TESTS_FILES_DIR}/sources/proxy.log.gz ${TESTS_FILES_DIR}/formats/proxy.log.format 0 ${TESTS_FILES_DIR}/sources/proxy.log)
if(MINESET_TESTS)
	add_inspector_compile_test(Tinendi_export_selection_to_mineset LINKS ${LINK_LIBRARIES} FILES export_selection_to_mineset.cpp)
	add_inspector_test(Tinendi_export_selection_to_mineset Tinendi_export_selection_to_mineset ${TESTS_FILES_DIR}/sources/proxy_mineset.log ${TESTS_FILES_DIR}/formats/proxy.log.format 0)
endif(MINESET_TESTS)

add_inspector_test(Tinendi_export_small_selection_gz Tinendi_export_selection ${TESTS_FILES_DIR}/sources/proxy.log.gz ${TESTS_FILES_DIR}/formats/proxy_small.log.format 0 ${TESTS_FILES_DIR}/sources/proxy_small.log)
add_inspector_test(Tinendi_export_small_selection_bz2 Tinendi_export_selection ${TESTS_FILES_DIR}/sources/proxy.log.bz2 ${TESTS_FILES_DIR}/formats/proxy_small.log.format 0 ${TESTS_FILES_DIR}/sources/proxy_small.log)
add_inspector_test(Tinendi_export_small_selection_zip Tinendi_export_selection ${TESTS_FILES_DIR}/sources/proxy.log.zip ${TESTS_FILES_DIR}/formats/proxy_small.log.format 0 ${TESTS_FILES_DIR}/sources/proxy_small.log)

# Test selection
add_inspector_test(Tinendi_export_odd_selection Tinendi_export_selection ${TESTS_FILES_DIR}/sources/proxy.log ${TESTS_FILES_DIR}/formats/proxy.log.format 1 ${TESTS_FILES_DIR}/sources/proxy_odd.log)

add_inspector_test(Tinendi_formatter_number Tinendi_export_selection ${TESTS_FILES_DIR}/picviz/formatter_number.csv ${TESTS_FILES_DIR}/picviz/formatter_number.csv.format 0)

add_inspector_compile_test(Tmapping_float_default LINKS ${LINK_LIBRARIES} FILES mapping_copy.cpp)
add_inspector_test(Tmapping_float_default Tmapping_float_default ${TESTS_FILES_DIR}/picviz/float_default_mapping.csv ${TESTS_FILES_DIR}/picviz/float_default_mapping.csv.format)
add_bench(Tmapping_float_default LINKS ${LINK_LIBRARIES} FILES mapping_copy.cpp)

add_inspector_compile_test(Tmapping_double_default LINKS ${LINK_LIBRARIES} FILES mapping_copy.cpp)
add_inspector_test(Tmapping_double_default Tmapping_double_default ${TESTS_FILES_DIR}/picviz/double_default_mapping.csv ${TESTS_FILES_DIR}/picviz/double_default_mapping.csv.format)
add_bench(Tmapping_double_default LINKS ${LINK_LIBRARIES} FILES mapping_copy.cpp)

add_inspector_compile_test(Tmapping_host_default LINKS ${LINK_LIBRARIES} FILES mapping_host_default.cpp)
add_inspector_test(Tmapping_host_default Tmapping_host_default)
add_bench(Tmapping_host_default LINKS ${LINK_LIBRARIES} FILES mapping_host_default.cpp)

add_inspector_compile_test(Tmapping_integer_default LINKS ${LINK_LIBRARIES} FILES mapping_copy.cpp)
add_inspector_test(Tmapping_integer_default Tmapping_integer_default ${TESTS_FILES_DIR}/picviz/integer_default_mapping.csv ${TESTS_FILES_DIR}/picviz/integer_default_mapping.csv.format)
add_bench(Tmapping_integer_default LINKS ${LINK_LIBRARIES} FILES mapping_copy.cpp)

add_inspector_compile_test(Tmapping_integer_unsigned LINKS ${LINK_LIBRARIES} FILES mapping_copy.cpp)
add_inspector_test(Tmapping_integer_unsigned Tmapping_integer_unsigned ${TESTS_FILES_DIR}/picviz/integer_unsigned_mapping.csv ${TESTS_FILES_DIR}/picviz/integer_unsigned_mapping.csv.format)
add_bench(Tmapping_integer_unsigned LINKS ${LINK_LIBRARIES} FILES mapping_copy.cpp)

add_inspector_compile_test(Tmapping_ipv4_default LINKS ${LINK_LIBRARIES} FILES mapping_copy.cpp)
add_inspector_test(Tmapping_ipv4_default Tmapping_ipv4_default ${TESTS_FILES_DIR}/picviz/ipv4_default_mapping.csv ${TESTS_FILES_DIR}/picviz/ipv4_default_mapping.csv.format)
add_bench(Tmapping_ipv4_default LINKS ${LINK_LIBRARIES} FILES mapping_copy.cpp)

add_inspector_compile_test(Tplotting_ipv4_uniform LINKS ${LINK_LIBRARIES} FILES plotting_uniform.cpp)
add_inspector_test(Tplotting_ipv4_uniform Tplotting_ipv4_uniform ${TESTS_FILES_DIR}/picviz/ipv4_default_mapping.csv ${TESTS_FILES_DIR}/picviz/ipv4_default_mapping.csv.format)
add_bench(Tplotting_ipv4_uniform LINKS ${LINK_LIBRARIES} FILES plotting_uniform.cpp)

add_inspector_compile_test(Tplotting_ipv6_uniform LINKS ${LINK_LIBRARIES} FILES plotting_uniform.cpp)
add_inspector_test(Tplotting_ipv6_uniform Tplotting_ipv6_uniform ${TESTS_FILES_DIR}/picviz/ipv6.csv ${TESTS_FILES_DIR}/picviz/ipv6.csv.format)
add_bench(Tplotting_ipv6_uniform LINKS ${LINK_LIBRARIES} FILES plotting_uniform.cpp)

add_inspector_compile_test(Tmapping_string_4bsort LINKS ${LINK_LIBRARIES} FILES mapping_string_4bsort.cpp)
add_inspector_test(Tmapping_string_4bsort Tmapping_string_4bsort)
add_bench(Tmapping_string_4bsort LINKS ${LINK_LIBRARIES} FILES mapping_string_4bsort.cpp)

add_inspector_compile_test(Tmapping_string_default LINKS ${LINK_LIBRARIES} FILES mapping_string_default.cpp)
add_inspector_test(Tmapping_string_default Tmapping_string_default)
add_bench(Tmapping_string_default LINKS ${LINK_LIBRARIES} FILES mapping_string_default.cpp)

add_inspector_compile_test(Tmapping_time_default LINKS ${LINK_LIBRARIES} FILES mapping_time_default.cpp)
add_inspector_test(Tmapping_time_default Tmapping_time_default)
add_bench(Tmapping_time_default LINKS ${LINK_LIBRARIES} FILES mapping_time_default.cpp)

add_inspector_compile_test(Tmapping_time_ms_default LINKS ${LINK_LIBRARIES} FILES mapping_time_ms_default.cpp)
add_inspector_test(Tmapping_time_ms_default Tmapping_time_ms_default)
add_bench(Tmapping_time_ms_default LINKS ${LINK_LIBRARIES} FILES mapping_time_ms_default.cpp)

add_inspector_compile_test(Tmapping_time_us_default LINKS ${LINK_LIBRARIES} FILES mapping_time_us_default.cpp)
add_inspector_test(Tmapping_time_us_default Tmapping_time_us_default)
add_bench(Tmapping_time_us_default LINKS ${LINK_LIBRARIES} FILES mapping_time_us_default.cpp)

add_inspector_compile_test(Tmapping_time_24h LINKS ${LINK_LIBRARIES} FILES mapping_time_24h.cpp)
add_inspector_test(Tmapping_time_24h Tmapping_time_24h)
add_bench(Tmapping_time_24h LINKS ${LINK_LIBRARIES} FILES mapping_time_24h.cpp)

add_inspector_compile_test(Tmapping_time_ms_24h LINKS ${LINK_LIBRARIES} FILES mapping_time_ms_24h.cpp)
add_inspector_test(Tmapping_time_ms_24h Tmapping_time_ms_24h)
add_bench(Tmapping_time_ms_24h LINKS ${LINK_LIBRARIES} FILES mapping_time_ms_24h.cpp)

add_inspector_compile_test(Tmapping_time_us_24h LINKS ${LINK_LIBRARIES} FILES mapping_time_us_24h.cpp)
add_inspector_test(Tmapping_time_us_24h Tmapping_time_us_24h)
add_bench(Tmapping_time_us_24h LINKS ${LINK_LIBRARIES} FILES mapping_time_us_24h.cpp)

add_inspector_compile_test(Tmapping_time_week LINKS ${LINK_LIBRARIES} FILES mapping_time_week.cpp)
add_inspector_test(Tmapping_time_week Tmapping_time_week)
add_bench(Tmapping_time_week LINKS ${LINK_LIBRARIES} FILES mapping_time_week.cpp)

add_inspector_compile_test(Tmapping_time_ms_week LINKS ${LINK_LIBRARIES} FILES mapping_time_ms_week.cpp)
add_inspector_test(Tmapping_time_ms_week Tmapping_time_ms_week)
add_bench(Tmapping_time_ms_week LINKS ${LINK_LIBRARIES} FILES mapping_time_ms_week.cpp)

add_inspector_compile_test(Tmapping_time_us_week LINKS ${LINK_LIBRARIES} FILES mapping_time_us_week.cpp)
add_inspector_test(Tmapping_time_us_week Tmapping_time_us_week)
add_bench(Tmapping_time_us_week LINKS ${LINK_LIBRARIES} FILES mapping_time_us_week.cpp)

add_inspector_compile_test(Tmapping_mac_address LINKS ${LINK_LIBRARIES} FILES mapping_mac_address.cpp)
add_inspector_test(Tmapping_mac_address_lin Tmapping_mac_address 2 ${TESTS_FILES_DIR}/picviz/mac_address_lin_mapping.format)
add_bench(Tmapping_mac_address_lin LINKS ${LINK_LIBRARIES} FILES mapping_mac_address.cpp)
add_inspector_test(Tmapping_mac_address_lin-uni Tmapping_mac_address 4 ${TESTS_FILES_DIR}/picviz/mac_address_lin-uni_mapping.format)
add_bench(Tmapping_mac_address_lin LINKS ${LINK_LIBRARIES} FILES mapping_mac_address.cpp)
add_inspector_test(Tmapping_mac_address_uni-lin Tmapping_mac_address 5 ${TESTS_FILES_DIR}/picviz/mac_address_uni-lin_mapping.format)
add_bench(Tmapping_mac_address_uni-lin LINKS ${LINK_LIBRARIES} FILES mapping_mac_address.cpp)
add_inspector_test(Tmapping_mac_address_uni-uni Tmapping_mac_address 6 ${TESTS_FILES_DIR}/picviz/mac_address_uni-uni_mapping.format)
add_bench(Tmapping_mac_address_uni-uni LINKS ${LINK_LIBRARIES} FILES mapping_mac_address.cpp)

add_inspector_compile_test(Tinendi_investigation LINKS ${LINK_LIBRARIES} FILES investigation.cpp)
add_inspector_test(Tinendi_investigation Tinendi_investigation)

add_inspector_compile_test(Tinendi_investigation_multisource LINKS ${LINK_LIBRARIES} FILES investigation_multisource.cpp)
add_inspector_test(Tinendi_investigation_multisource Tinendi_investigation_multisource)

add_inspector_compile_test(Tplotting_integer_default_default LINKS ${LINK_LIBRARIES} FILES plotting.cpp)
add_inspector_test(Tplotting_integer_default_default Tplotting_integer_default_default ${TESTS_FILES_DIR}/picviz/integer_default_mapping.csv ${TESTS_FILES_DIR}/picviz/integer_default_mapping.csv.format)
add_bench(Tplotting_integer_default_default LINKS ${LINK_LIBRARIES} FILES plotting.cpp)

add_inspector_compile_test(Tplotting_integer_unsigned_default LINKS ${LINK_LIBRARIES} FILES plotting.cpp)
add_inspector_test(Tplotting_integer_unsigned_default Tplotting_integer_unsigned_default ${TESTS_FILES_DIR}/picviz/integer_unsigned_mapping.csv ${TESTS_FILES_DIR}/picviz/integer_unsigned_mapping.csv.format)
add_bench(Tplotting_integer_unsigned_default LINKS ${LINK_LIBRARIES} FILES plotting.cpp)

add_inspector_compile_test(Tplotting_float_default_default LINKS ${LINK_LIBRARIES} FILES plotting.cpp)
add_inspector_test(Tplotting_float_default_default Tplotting_float_default_default ${TESTS_FILES_DIR}/picviz/float_default_mapping.csv ${TESTS_FILES_DIR}/picviz/float_default_mapping.csv.format)
add_bench(Tplotting_float_default_default LINKS ${LINK_LIBRARIES} FILES plotting.cpp)

add_inspector_compile_test(Tplotting_double_default_default LINKS ${LINK_LIBRARIES} FILES plotting.cpp)
add_inspector_test(Tplotting_double_default_default Tplotting_double_default_default ${TESTS_FILES_DIR}/picviz/double_default_mapping.csv ${TESTS_FILES_DIR}/picviz/double_default_mapping.csv.format)
add_bench(Tplotting_double_default_default LINKS ${LINK_LIBRARIES} FILES plotting.cpp)

add_inspector_compile_test(Tplotting_string_default_enum LINKS ${LINK_LIBRARIES} FILES plotting_string_default_enum.cpp)
add_inspector_test(Tplotting_string_default_enum Tplotting_string_default_enum)
add_bench(Tplotting_string_default_enum LINKS ${LINK_LIBRARIES} FILES plotting_string_default_enum.cpp)

add_inspector_compile_test(Tplotting_time_24h_minmax LINKS ${LINK_LIBRARIES} FILES plotting_time_24h_minmax.cpp)
add_inspector_test(Tplotting_time_24h_minmax Tplotting_time_24h_minmax)
add_bench(Tplotting_time_24h_minmax LINKS ${LINK_LIBRARIES} FILES plotting_time_24h_minmax.cpp)

add_inspector_compile_test(Tplotting_time_week_minmax LINKS ${LINK_LIBRARIES} FILES plotting_time_week_minmax.cpp)
add_inspector_test(Tplotting_time_week_minmax Tplotting_time_week_minmax)
add_bench(Tplotting_time_week_minmax LINKS ${LINK_LIBRARIES} FILES plotting_time_week_minmax.cpp)

add_inspector_compile_test(Tplotting_integer_default_log LINKS ${LINK_LIBRARIES} FILES plotting.cpp)
add_inspector_test(Tplotting_integer_default_log Tplotting_integer_default_log ${TESTS_FILES_DIR}/picviz/integer_default_mapping.csv ${TESTS_FILES_DIR}/picviz/integer_default_mapping.csv.format)
add_bench(Tplotting_integer_default_log LINKS ${LINK_LIBRARIES} FILES plotting.cpp)

add_inspector_compile_test(Tplotting_integer_unsigned_log LINKS ${LINK_LIBRARIES} FILES plotting.cpp)
add_inspector_test(Tplotting_integer_unsigned_log Tplotting_integer_unsigned_log ${TESTS_FILES_DIR}/picviz/integer_unsigned_mapping.csv ${TESTS_FILES_DIR}/picviz/integer_unsigned_mapping.csv.format)
add_bench(Tplotting_integer_unsigned_log LINKS ${LINK_LIBRARIES} FILES plotting.cpp)

add_inspector_compile_test(Tplotting_float_default_log LINKS ${LINK_LIBRARIES} FILES plotting.cpp)
add_inspector_test(Tplotting_float_default_log Tplotting_float_default_log ${TESTS_FILES_DIR}/picviz/float_default_mapping.csv ${TESTS_FILES_DIR}/picviz/float_default_mapping.csv.format)
add_bench(Tplotting_float_default_log LINKS ${LINK_LIBRARIES} FILES plotting.cpp)

add_inspector_compile_test(Tplotting_time_24h_log LINKS ${LINK_LIBRARIES} FILES plotting_time_24h_log.cpp)
add_inspector_test(Tplotting_time_24h_log Tplotting_time_24h_log)
add_bench(Tplotting_time_24h_log LINKS ${LINK_LIBRARIES} FILES plotting_time_24h_log.cpp)

add_inspector_compile_test(Tplotting_time_week_log LINKS ${LINK_LIBRARIES} FILES plotting_time_week_log.cpp)
add_inspector_test(Tplotting_time_week_log Tplotting_time_week_log)
add_bench(Tplotting_time_week_log LINKS ${LINK_LIBRARIES} FILES plotting_time_week_log.cpp)

add_inspector_compile_test(Tplotting_port LINKS ${LINK_LIBRARIES} FILES plotting_port.cpp)
add_inspector_test(Tplotting_port Tplotting_port)
add_bench(Tplotting_port LINKS ${LINK_LIBRARIES} FILES plotting_port.cpp)

add_inspector_compile_test(Tplotting_integer_default_enum LINKS ${LINK_LIBRARIES} FILES plotting.cpp)
add_inspector_test(Tplotting_integer_default_enum Tplotting_integer_default_enum ${TESTS_FILES_DIR}/picviz/integer_default_mapping.csv ${TESTS_FILES_DIR}/picviz/integer_default_enum.csv.format)
add_bench(Tplotting_integer_default_enum LINKS ${LINK_LIBRARIES} FILES plotting.cpp)

# FIXME : This test is a bit slow but it test a lot of thing. We should split
# it in multiple tests
add_inspector_compile_test(TPVSelection LINKS ${LINK_LIBRARIES} FILES PVSelection.cpp)
add_inspector_test(TPVSelection TPVSelection)

add_inspector_compile_test(Tinendi_bad_convertions_merge LINKS ${LINK_LIBRARIES} FILES bad_convertions_merge.cpp)
add_inspector_test(Tinendi_bad_convertions_merge Tinendi_bad_convertions_merge)

add_inspector_compile_test(Tinendi_correlation LINKS ${LINK_LIBRARIES} FILES correlation.cpp)
add_inspector_test(Tinendi_correlation Tinendi_correlation)
add_bench(Tinendi_correlation LINKS ${LINK_LIBRARIES} FILES correlation.cpp)

add_inspector_compile_test(Trush_multi_chunk LINKS ${LINK_LIBRARIES} FILES multi_chunk.cpp)
add_inspector_test(Trush_multi_chunk Trush_multi_chunk)

add_inspector_compile_test(Trush_multi_chunk_header LINKS ${LINK_LIBRARIES} FILES multi_chunk_header.cpp)
add_inspector_test(Trush_multi_chunk_header Trush_multi_chunk_header)

add_inspector_compile_test(Tinendi_export_plotted LINKS ${LINK_LIBRARIES} FILES export_plotted.cpp)
add_inspector_test(Tinendi_export_plotted Tinendi_export_plotted ${TESTS_FILES_DIR}/picviz/export_plotted.csv.in ${TESTS_FILES_DIR}/picviz/export_plotted.csv.out ${TESTS_FILES_DIR}/picviz/export_plotted.csv.format)
add_bench(Tinendi_export_plotted LINKS ${LINK_LIBRARIES} FILES export_plotted.cpp)

add_inspector_compile_test(Tinendi_plotting_one_value LINKS ${LINK_LIBRARIES} FILES plotting_one_value.cpp)
add_inspector_test(Tinendi_plotting_one_value Tinendi_plotting_one_value ${TESTS_FILES_DIR}/picviz/plotting_one_value.csv ${TESTS_FILES_DIR}/picviz/plotting_one_value.csv.format)

add_inspector_compile_test(Tinendi_axis_gradient LINKS ${LINK_LIBRARIES} FILES axis_gradient.cpp)
add_inspector_test(Tinendi_axis_gradient Tinendi_axis_gradient ${TESTS_FILES_DIR}/picviz/axis_gradient.csv ${TESTS_FILES_DIR}/picviz/axis_gradient.csv.format)
add_bench(Tinendi_axis_gradient LINKS ${LINK_LIBRARIES} FILES axis_gradient.cpp)

add_inspector_compile_test(Tinendi_axes_combination LINKS ${LINK_LIBRARIES} FILES axes_combination.cpp)
add_inspector_test(Tinendi_axes_combination Tinendi_axes_combination)

add_inspector_compile_test(Tinendi_view_unselected_zombie_toggle LINKS ${LINK_LIBRARIES} FILES view_unselected_zombie_toggle.cpp)
add_inspector_test(Tinendi_view_unselected_zombie_toggle Tinendi_view_unselected_zombie_toggle)

# Cancel import
add_inspector_compile_test(Tinendi_cancel_import LINKS ${LINK_LIBRARIES} FILES cancel_import.cpp)
add_inspector_test(Tinendi_cancel_import Tinendi_cancel_import ${TESTS_FILES_DIR}/sources/proxy.log.gz ${TESTS_FILES_DIR}/formats/proxy.log.format)

# Cancel export
add_inspector_compile_test(Tinendi_cancel_export LINKS ${LINK_LIBRARIES} FILES cancel_export.cpp)
add_inspector_test(Tinendi_cancel_export Tinendi_cancel_export ${TESTS_FILES_DIR}/sources/proxy.log.gz ${TESTS_FILES_DIR}/formats/proxy.log.format)

# Range subsampler
add_inspector_compile_test(Tinendi_range_subsampler LINKS ${LINK_LIBRARIES} FILES range_subsampler.cpp)
add_inspector_test(Tinendi_range_subsampler Tinendi_range_subsampler)

# multi-inputs
add_inspector_compile_test(Tinendi_multi_inputs LINKS ${LINK_LIBRARIES} FILES multi_inputs.cpp)
add_inspector_test(Tinendi_multi_inputs Tinendi_multi_inputs)

# python_scripting
add_inspector_compile_test(Tinendi_python_scripting LINKS ${LINK_LIBRARIES} FILES python_scripting.cpp)
add_inspector_test(Tinendi_python_scripting Tinendi_python_scripting ${TESTS_FILES_DIR}/sources/proxy_sample1.log ${TESTS_FILES_DIR}/sources/proxy_sample1_with_python_script.log.format ${TESTS_FILES_DIR}/exports/Tinendi_python_scripting)
add_inspector_test(Tinendi_embeded_python_scripting Tinendi_python_scripting ${TESTS_FILES_DIR}/sources/proxy_sample1.log ${TESTS_FILES_DIR}/sources/proxy_sample1_with_embeded_python_script.log.format ${TESTS_FILES_DIR}/exports/Tinendi_python_scripting)