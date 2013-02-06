#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

#add_definitions(-ftree-vectorizer-verbose=4)

# if(WIN32)
# 	set(PVCORE_LIBRARY pvcore)
# else(WIN32)
# 	set(PVCORE_LIBRARY pvcore m)
# endif(WIN32)

declare_pv_test(Tcore_capslock "capslock.cpp" "${QT_LIBRARIES};pvkernel")
add_test(TestCapslock Tcore_capslock)

declare_pv_test(Tcore_simple "simple.cpp" "pvkernel;${QT_LIBRARIES}")
add_test(TestSimple Tcore_simple)

#add_executable(Tcore_debug debug.cpp)
#target_link_libraries(Tcore_debug pvkernel ${QT_LIBRARIES})
#add_test(TestDebug Tcore_debug)

declare_pv_test(Tcore_network "network.cpp" "pvkernel;${QT_LIBRARIES}")
add_test(TestNetwork Tcore_network)

# add_executable(Tcore_process process.cpp)
# target_link_libraries(Tcore_process ${PVLIBRARIES} pthread)
# add_test(TestProcess Tcore_process)

declare_pv_test(Tcore_sysinfo "sysinfo.cpp" "pvkernel;${QT_LIBRARIES}")
add_test(TestSysinfo Tcore_sysinfo)

declare_pv_test(Tcore_logger "logger.cpp" "pvkernel;${QT_LIBRARIES}")
add_test(TestLogger Tcore_logger)

# Time format parser test
declare_pv_test(Tcore_timeformat "time-format.cpp" "pvkernel;${QT_LIBRARIES};${ICU_LIBRARY}")
add_test(TestTimeformat Tcore_timeformat ${CMAKE_CURRENT_SOURCE_DIR}/test-files/time_format)

# Time format performance
declare_pv_test(Tcore_timeformat_perf "time-format-perf.cpp" "pvkernel;${QT_LIBRARIES};${ICU_LIBRARY}")
target_link_libraries(Tcore_timeformat_perf pvkernel ${QT_LIBRARIES} ${ICU_LIBRARY})

# Serialization
declare_pv_test(Tcore_serialize "serialize.cpp" "pvkernel;${QT_LIBRARIES}")

declare_pv_test(Tcore_argument_serialize "argument_serialize.cpp" "pvkernel;${QT_LIBRARIES}")
add_test(TestArgumentSerialize Tcore_argument_serialize)

# Strings
declare_pv_test(Tcore_pvstring pvstring.cpp "pvkernel;${QT_LIBRARIES};${ICU_LIBRARY}")

# Matrix
declare_pv_test(Tcore_matrix matrix.cpp "pvkernel;${QT_LIBRARIES}")

# OpenGL's fill vbo bench
declare_pv_test(Tcore_vbo vbo.cpp "pvkernel;${QT_LIBRARIES}")

declare_pv_test(Tcore_podtree "pod_tree.cpp" "pvkernel;${QT_LIBRARIES}")

declare_pv_test(Tcore_sizes "sizes.cpp" "pvkernel;${QT_LIBRARIES}")

declare_pv_test(Tcore_stable_reverse "stable_reverse.cpp" "pvkernel;${QT_LIBRARIES}")

declare_pv_test(Tcore_bench_stable_reverse "bench_stable_reverse.cpp" "pvkernel;${QT_LIBRARIES}")

declare_pv_test(Tcore_launch_adaptive "launch_adaptive.cpp" "pvkernel;${QT_LIBRARIES};${Boost_LIBRARIES}")

declare_pv_test(Tcore_transp "transp.cpp" "pvkernel;${QT_LIBRARIES}")

# Data tree
declare_pv_test(Tcore_datatree "datatree.cpp" "pvkernel;${QT_LIBRARIES}")
add_test(TestDataTree Tcore_datatree)

# Shared Pointer
#add_executable(Tcore_shared_ptr shared_ptr.cpp)
#target_link_libraries(Tcore_shared_ptr pvkernel)

# Weak Pointer
#add_executable(Tcore_weak_ptr weak_ptr.cpp)
#target_link_libraries(Tcore_weak_ptr pvkernel)

# Function traits
declare_pv_test(Tcore_function_traits "function_traits.cpp" "pvkernel")

# Const forward
declare_pv_test(Tcore_const_fwd "const_fwd.cpp" "pvkernel")

# Bit count
declare_pv_test(Tcore_bit_count "bit_count.cpp" "pvkernel;${QT_LIBRARIES};${OpenMP_LIBRARIES}")
add_test(TestBitCount Tcore_bit_count 65535)

# Bit visitor
declare_pv_test(Tcore_visit_bits "visit_bits.cpp" "pvkernel;${QT_LIBRARIES};${OpenMP_LIBRARIES}")
add_test(TestVisitBits Tcore_visit_bits)

# Bit field
declare_pv_test(Tcore_bit_field_visit "bit_field_visit.cpp" "pvkernel;${QT_LIBRARIES};${OpenMP_LIBRARIES}")
add_test(TestBitFieldVisit Tcore_bit_field_visit)

# POD tree
declare_pv_test(Tcore_pod_tree "pod_tree.cpp" "pvkernel;${QT_LIBRARIES}")

# Hardware concurrency
declare_pv_test(Tcore_hardware_concurrency "hardware_concurrency.cpp" "pvkernel;${QT_LIBRARIES}")

# Shared ptr forward
declare_pv_test(Tcore_shared_ptr_fwd "shared_ptr_fwd.cpp" "pvkernel")

# Raid0
declare_pv_test(Tcore_raid_test "raid_test.cpp" "pvkernel")

# Decimal storage
declare_pv_test(Tcore_decimal_storage "decimal_storage.cpp" "pvkernel")
add_test(TestDecimalStorage Tcore_decimal_storage)

# Huge POD vector
declare_pv_test(Tcore_huge_vector "huge_vector.cpp" "pvkernel")

# lambda_connect
declare_pv_test(Tcore_lambda_connect "lambda_connect.cpp" "pvkernel;${QT_LIBRARIES}")
