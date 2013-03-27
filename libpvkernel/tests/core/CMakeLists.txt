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

declare_pv_test(Tcore_capslock capslock.cpp)
target_link_libraries(Tcore_capslock ${QT_LIBRARIES} pvkernel)
add_test(TestCapslock Tcore_capslock)

declare_pv_test(Tcore_simple simple.cpp)
target_link_libraries(Tcore_simple pvkernel ${QT_LIBRARIES})
add_test(TestSimple Tcore_simple)

#add_executable(Tcore_debug debug.cpp)
#target_link_libraries(Tcore_debug pvkernel ${QT_LIBRARIES})
#add_test(TestDebug Tcore_debug)

declare_pv_test(Tcore_network network.cpp)
target_link_libraries(Tcore_network pvkernel ${QT_LIBRARIES})
add_test(TestNetwork Tcore_network)

# add_executable(Tcore_process process.cpp)
# target_link_libraries(Tcore_process ${PVLIBRARIES} pthread)
# add_test(TestProcess Tcore_process)

declare_pv_test(Tcore_sysinfo sysinfo.cpp)
target_link_libraries(Tcore_sysinfo pvkernel ${QT_LIBRARIES})
add_test(TestSysinfo Tcore_sysinfo)

declare_pv_test(Tcore_logger logger.cpp)
target_link_libraries(Tcore_logger pvkernel ${QT_LIBRARIES})
add_test(TestLogger Tcore_logger)

# Time format parser test
declare_pv_test(Tcore_timeformat time-format.cpp)
target_link_libraries(Tcore_timeformat pvkernel ${QT_LIBRARIES} ${ICU_LIBRARY})
add_test(TestTimeformat Tcore_timeformat ${CMAKE_CURRENT_SOURCE_DIR}/test-files/time_format)

# Time format performance
declare_pv_test(Tcore_timeformat_perf time-format-perf.cpp)
target_link_libraries(Tcore_timeformat_perf pvkernel ${QT_LIBRARIES} ${ICU_LIBRARY})

# Serialization
declare_pv_test(Tcore_serialize serialize.cpp)
target_link_libraries(Tcore_serialize pvkernel ${QT_LIBRARIES})

declare_pv_test(Tcore_argument_serialize argument_serialize.cpp)
target_link_libraries(Tcore_argument_serialize pvkernel ${QT_LIBRARIES})
add_test(TestArgumentSerialize Tcore_argument_serialize)

# Strings
declare_pv_test(Tcore_pvstring pvstring.cpp)
target_link_libraries(Tcore_pvstring pvkernel ${QT_LIBRARIES} ${ICU_LIBRARY})

# Matrix
declare_pv_test(Tcore_matrix matrix.cpp)
target_link_libraries(Tcore_matrix pvkernel ${QT_LIBRARIES})

# OpenGL's fill vbo bench
declare_pv_test(Tcore_vbo vbo.cpp)
target_link_libraries(Tcore_vbo pvkernel ${QT_LIBRARIES})

declare_pv_test(Tcore_podtree pod_tree.cpp)
target_link_libraries(Tcore_podtree pvkernel ${QT_LIBRARIES})

declare_pv_test(Tcore_sizes sizes.cpp)
target_link_libraries(Tcore_sizes pvkernel ${QT_LIBRARIES})

declare_pv_test(Tcore_stable_reverse stable_reverse.cpp)
target_link_libraries(Tcore_stable_reverse pvkernel ${QT_LIBRARIES})

declare_pv_test(Tcore_bench_stable_reverse bench_stable_reverse.cpp)
target_link_libraries(Tcore_bench_stable_reverse pvkernel ${QT_LIBRARIES})

declare_pv_test(Tcore_launch_adaptive launch_adaptive.cpp)
target_link_libraries(Tcore_launch_adaptive pvkernel ${QT_LIBRARIES} ${Boost_LIBRARIES})

declare_pv_test(Tcore_transp transp.cpp)
target_link_libraries(Tcore_transp pvkernel ${QT_LIBRARIES})

# Data tree
declare_pv_test(Tcore_datatree datatree.cpp)
target_link_libraries(Tcore_datatree pvkernel ${QT_LIBRARIES})
add_test(TestDataTree Tcore_datatree)

# Shared Pointer
declare_pv_test(Tcore_shared_ptr shared_ptr.cpp)
target_link_libraries(Tcore_shared_ptr pvkernel)

# Weak Pointer
declare_pv_test(Tcore_weak_ptr weak_ptr.cpp)
target_link_libraries(Tcore_weak_ptr pvkernel)

# Function traits
declare_pv_test(Tcore_function_traits function_traits.cpp)
target_link_libraries(Tcore_function_traits pvkernel)

# Const forward
declare_pv_test(Tcore_const_fwd const_fwd.cpp)
target_link_libraries(Tcore_const_fwd pvkernel)

# Bit count
declare_pv_test(Tcore_bit_count bit_count.cpp)
target_link_libraries(Tcore_bit_count pvkernel ${QT_LIBRARIES} ${OpenMP_LIBRARIES})
add_test(TestBitCount Tcore_bit_count 65535)

# Bit visitor
declare_pv_test(Tcore_visit_bits visit_bits.cpp)
target_link_libraries(Tcore_visit_bits pvkernel ${QT_LIBRARIES} ${OpenMP_LIBRARIES})
add_test(TestVisitBits Tcore_visit_bits)

# Byte visitor performances 
declare_pv_test(Tcore_visit_bytes_perf visit_bytes_perf.cpp)
target_link_libraries(Tcore_visit_bytes_perf pvkernel ${QT_LIBRARIES} ${OpenMP_LIBRARIES})
add_test(TestVisitBytesPerf Tcore_visit_bytes_perf)

# Bit field
declare_pv_test(Tcore_bit_field_visit bit_field_visit.cpp)
target_link_libraries(Tcore_bit_field_visit pvkernel ${QT_LIBRARIES} ${OpenMP_LIBRARIES})
add_test(TestBitFieldVisit Tcore_bit_field_visit)
declare_pv_test(Tcore_bit_field_visit_packed bit_field_visit_packed.cpp)
target_link_libraries(Tcore_bit_field_visit_packed pvkernel ${QT_LIBRARIES} ${OpenMP_LIBRARIES})
add_test(TestBitFieldVisitPacked Tcore_bit_field_visit_packed)

# POD tree
declare_pv_test(Tcore_pod_tree pod_tree.cpp)
target_link_libraries(Tcore_pod_tree pvkernel ${QT_LIBRARIES})

# file helper
declare_pv_test(Tcore_file_helper file_helper.cpp)
target_link_libraries(Tcore_file_helper pvkernel)
add_test(TestFileHelper Tcore_file_helper)

# Hardware concurrency
declare_pv_test(Tcore_hardware_concurrency hardware_concurrency.cpp)
target_link_libraries(Tcore_hardware_concurrency pvkernel ${QT_LIBRARIES})

# Shared ptr forward
declare_pv_test(Tcore_shared_ptr_fwd shared_ptr_fwd.cpp)
target_link_libraries(Tcore_shared_ptr_fwd pvkernel)

# Raid0
declare_pv_test(Tcore_raid_test raid_test.cpp)
target_link_libraries(Tcore_raid_test pvkernel)

# Decimal storage
declare_pv_test(Tcore_decimal_storage decimal_storage.cpp)
target_link_libraries(Tcore_decimal_storage pvkernel)
add_test(TestDecimalStorage Tcore_decimal_storage)

# Huge POD vector
declare_pv_test(Tcore_huge_vector huge_vector.cpp)
target_link_libraries(Tcore_huge_vector pvkernel)

# lambda_connect
declare_pv_test(Tcore_lambda_connect lambda_connect.cpp)
target_link_libraries(Tcore_lambda_connect pvkernel ${QT_LIBRARIES})

# Intrinsics
declare_pv_test(Tcore_intrinsics intrinsics.cpp)
target_link_libraries(Tcore_intrinsics pvkernel)
add_test(TestIntrinsics Tcore_intrinsics)

declare_lib_tests_target()
