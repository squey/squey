#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

add_inspector_compile_test(Tcore_file_helper LINKS ${LINK_LIBRARIES} FILES file_helper.cpp)
add_inspector_test(Tcore_file_helper Tcore_file_helper)

add_inspector_compile_test(Tcore_intrinsics LINKS ${LINK_LIBRARIES} FILES intrinsics.cpp)
add_inspector_test(Tcore_intrinsics Tcore_intrinsics)

add_inspector_compile_test(Tcore_visit_bits LINKS ${LINK_LIBRARIES} ${OpenMP_LIBRARIES} FILES visit_bits.cpp)
add_inspector_test(Tcore_visit_bits Tcore_visit_bits)

add_inspector_compile_test(Tcore_bit_field_visit LINKS ${LINK_LIBRARIES} ${OpenMP_LIBRARIES} FILES bit_field_visit.cpp)
add_inspector_test(Tcore_bit_field_visit Tcore_bit_field_visit)

add_inspector_compile_test(Tcore_argument_serialize LINKS ${LINK_LIBRARIES} FILES argument_serialize.cpp)
add_inspector_test(Tcore_argument_serialize Tcore_argument_serialize)

add_inspector_compile_test(Tcore_network LINKS ${LINK_LIBRARIES} FILES network.cpp)
add_inspector_test(Tcore_network Tcore_network)

add_inspector_compile_test(Tcore_timeformat LINKS ${LINK_LIBRARIES} ${ICU_LIBRARY} FILES time-format.cpp)
add_inspector_test(Tcore_timeformat Tcore_timeformat ${TESTS_FILES_DIR}/pvkernel/core/time_format)
