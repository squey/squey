#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

add_pv_compile_test(Tcore_timeformat_perf LINKS pvkernel ${ICU_LIBRARY} FILES time-format-perf.cpp)
add_pv_compile_test(Tcore_stable_reverse LINKS pvkernel FILES stable_reverse.cpp)
add_pv_compile_test(Tcore_bench_stable_reverse LINKS pvkernel FILES bench_stable_reverse.cpp)
add_pv_compile_test(Tcore_hardware_concurrency LINKS pvkernel FILES hardware_concurrency.cpp)
add_pv_compile_test(Tcore_huge_vector LINKS pvkernel FILES huge_vector.cpp)

add_pv_compile_test(Tcore_file_helper LINKS pvkernel FILES file_helper.cpp)
add_test(Tcore_file_helper Tcore_file_helper)

add_pv_compile_test(Tcore_intrinsics LINKS pvkernel FILES intrinsics.cpp)
add_test(Tcore_intrinsics Tcore_intrinsics)

add_pv_compile_test(Tcore_visit_bits LINKS pvkernel ${OpenMP_LIBRARIES} FILES visit_bits.cpp)
add_test(Tcore_visit_bits Tcore_visit_bits)

add_pv_compile_test(Tcore_bit_field_visit LINKS pvkernel ${OpenMP_LIBRARIES} FILES bit_field_visit.cpp)
add_test(Tcore_bit_field_visit Tcore_bit_field_visit)

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

