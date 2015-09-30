#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012
# Copyright (C) ESI Group INENDI 2015

add_pv_compile_test(Tcore_timeformat_perf LINKS pvkernel ${ICU_LIBRARY} FILES time-format-perf.cpp)
add_pv_compile_test(Tcore_serialize LINKS pvkernel FILES serialize.cpp)
add_pv_compile_test(Tcore_pvstring LINKS pvkernel ${ICU_LIBRARY} FILES pvstring.cpp)
add_pv_compile_test(Tcore_matrix LINKS pvkernel FILES matrix.cpp)
add_pv_compile_test(Tcore_vbo LINKS pvkernel FILES vbo.cpp)
add_pv_compile_test(Tcore_podtree LINKS pvkernel FILES pod_tree.cpp)
add_pv_compile_test(Tcore_sizes LINKS pvkernel FILES sizes.cpp)
add_pv_compile_test(Tcore_stable_reverse LINKS pvkernel FILES stable_reverse.cpp)
add_pv_compile_test(Tcore_bench_stable_reverse LINKS pvkernel FILES bench_stable_reverse.cpp)
add_pv_compile_test(Tcore_launch_adaptive LINKS pvkernel ${Boost_LIBRARIES} FILES launch_adaptive.cpp)
add_pv_compile_test(Tcore_transp LINKS pvkernel FILES transp.cpp)
add_pv_compile_test(Tcore_shared_ptr LINKS pvkernel FILES shared_ptr.cpp)
add_pv_compile_test(Tcore_weak_ptr LINKS pvkernel FILES weak_ptr.cpp)
add_pv_compile_test(Tcore_function_traits LINKS pvkernel FILES function_traits.cpp)
add_pv_compile_test(Tcore_const_fwd LINKS pvkernel FILES const_fwd.cpp)
add_pv_compile_test(Tcore_pod_tree LINKS pvkernel FILES pod_tree.cpp)
add_pv_compile_test(Tcore_hardware_concurrency LINKS pvkernel FILES hardware_concurrency.cpp)
add_pv_compile_test(Tcore_shared_ptr_fwd LINKS pvkernel FILES shared_ptr_fwd.cpp)
add_pv_compile_test(Tcore_raid_test LINKS pvkernel FILES raid_test.cpp)
add_pv_compile_test(Tcore_huge_vector LINKS pvkernel FILES huge_vector.cpp)
add_pv_compile_test(Tcore_lambda_connect LINKS pvkernel FILES lambda_connect.cpp)
add_pv_compile_test(Tcore_memory2d LINKS pvkernel FILES memory2d.cpp)

add_pv_compile_test(Tcore_file_helper LINKS pvkernel FILES file_helper.cpp)
add_test(Tcore_file_helper Tcore_file_helper)

add_pv_compile_test(Tcore_decimal_storage LINKS pvkernel FILES decimal_storage.cpp)
add_test(Tcore_decimal_storage Tcore_decimal_storage 1000)

add_pv_compile_test(Tcore_intrinsics LINKS pvkernel FILES intrinsics.cpp)
add_test(Tcore_intrinsics Tcore_intrinsics)

add_pv_compile_test(Tcore_bit_count LINKS pvkernel ${OpenMP_LIBRARIES} FILES bit_count.cpp)
add_test(Tcore_bit_count Tcore_bit_count 65535)

add_pv_compile_test(Tcore_visit_bits LINKS pvkernel ${OpenMP_LIBRARIES} FILES visit_bits.cpp)
add_test(Tcore_visit_bits Tcore_visit_bits)

add_pv_compile_test(Tcore_visit_bytes_perf LINKS pvkernel ${OpenMP_LIBRARIES} FILES visit_bytes_perf.cpp)
add_test(Tcore_visit_bytes_perf Tcore_visit_bytes_perf 300)

add_pv_compile_test(Tcore_bit_field_visit LINKS pvkernel ${OpenMP_LIBRARIES} FILES bit_field_visit.cpp)
add_test(Tcore_bit_field_visit Tcore_bit_field_visit)

add_pv_compile_test(Tcore_bit_field_visit_packed LINKS pvkernel ${OpenMP_LIBRARIES} FILES bit_field_visit_packed.cpp)
add_test(Tcore_bit_field_visit_packed Tcore_bit_field_visit_packed)

add_pv_compile_test(Tcore_datatree LINKS pvkernel FILES datatree.cpp)
add_test(Tcore_datatree Tcore_datatree)

add_pv_compile_test(Tcore_argument_serialize LINKS pvkernel FILES argument_serialize.cpp)
add_test(Tcore_argument_serialize Tcore_argument_serialize)

add_pv_compile_test(Tcore_simple LINKS pvkernel FILES simple.cpp)
add_test(Tcore_simple Tcore_simple)

add_pv_compile_test(Tcore_network LINKS pvkernel FILES network.cpp)
add_test(Tcore_network Tcore_network)

add_pv_compile_test(Tcore_logger LINKS pvkernel FILES logger.cpp)
add_test(Tcore_logger  Tcore_logger)

add_pv_compile_test(Tcore_timeformat LINKS pvkernel ${ICU_LIBRARY} FILES time-format.cpp)
add_test(Tcore_timeformat Tcore_timeformat ${TESTS_FILES_DIR}/pvkernel/core/time_format)
