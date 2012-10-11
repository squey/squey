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

add_executable(Tcore_capslock capslock.cpp)
target_link_libraries(Tcore_capslock pvkernel ${QT_LIBRARIES})
add_test(TestCapslock Tcore_capslock)

add_executable(Tcore_simple simple.cpp)
target_link_libraries(Tcore_simple pvkernel ${QT_LIBRARIES})
add_test(TestSimple Tcore_simple)

#add_executable(Tcore_debug debug.cpp)
#target_link_libraries(Tcore_debug pvkernel ${QT_LIBRARIES})
#add_test(TestDebug Tcore_debug)

add_executable(Tcore_network network.cpp)
target_link_libraries(Tcore_network pvkernel ${QT_LIBRARIES})
add_test(TestNetwork Tcore_network)

# add_executable(Tcore_process process.cpp)
# target_link_libraries(Tcore_process ${PVLIBRARIES} pthread)
# add_test(TestProcess Tcore_process)

add_executable(Tcore_sysinfo sysinfo.cpp)
target_link_libraries(Tcore_sysinfo pvkernel ${QT_LIBRARIES})
add_test(TestSysinfo Tcore_sysinfo)

add_executable(Tcore_logger logger.cpp)
target_link_libraries(Tcore_logger pvkernel ${QT_LIBRARIES})
add_test(TestLogger Tcore_logger)

# Time format parser test
add_executable(Tcore_timeformat time-format.cpp)
target_link_libraries(Tcore_timeformat pvkernel ${QT_LIBRARIES} ${ICU_LIBRARY})
add_test(TestTimeformat Tcore_timeformat test-files/time_format)

# Time format performance
add_executable(Tcore_timeformat_perf time-format-perf.cpp)
target_link_libraries(Tcore_timeformat_perf pvkernel ${QT_LIBRARIES} ${ICU_LIBRARY})

# Serialization
add_executable(Tcore_serialize serialize.cpp)
target_link_libraries(Tcore_serialize pvkernel ${QT_LIBRARIES})

add_executable(Tcore_argument_serialize argument_serialize.cpp)
target_link_libraries(Tcore_argument_serialize pvkernel ${QT_LIBRARIES})
add_test(TestArgumentSerialize Tcore_argument_serialize)

# Strings
add_executable(Tcore_pvstring pvstring.cpp)
target_link_libraries(Tcore_pvstring pvkernel ${QT_LIBRARIES} ${ICU_LIBRARY})

# Matrix
add_executable(Tcore_matrix matrix.cpp)
target_link_libraries(Tcore_matrix pvkernel ${QT_LIBRARIES})

# OpenGL's fill vbo bench
add_executable(Tcore_vbo vbo.cpp)
target_link_libraries(Tcore_vbo pvkernel ${QT_LIBRARIES})

add_executable(Tcore_sizes sizes.cpp)
target_link_libraries(Tcore_sizes pvkernel ${QT_LIBRARIES})

add_executable(Tcore_stable_reverse stable_reverse.cpp)
target_link_libraries(Tcore_stable_reverse pvkernel ${QT_LIBRARIES})

add_executable(Tcore_bench_stable_reverse bench_stable_reverse.cpp)
target_link_libraries(Tcore_bench_stable_reverse pvkernel ${QT_LIBRARIES})

add_executable(Tcore_launch_adaptive launch_adaptive.cpp)
target_link_libraries(Tcore_launch_adaptive pvkernel ${QT_LIBRARIES} ${Boost_LIBRARIES})

add_executable(Tcore_transp transp.cpp)
target_link_libraries(Tcore_transp pvkernel ${QT_LIBRARIES})

# Data tree
add_executable(Tcore_datatree datatree.cpp)
target_link_libraries(Tcore_datatree pvkernel ${QT_LIBRARIES})

# Shared Pointer
#add_executable(Tcore_shared_ptr shared_ptr.cpp)
#target_link_libraries(Tcore_shared_ptr pvkernel)

# Weak Pointer
#add_executable(Tcore_weak_ptr weak_ptr.cpp)
#target_link_libraries(Tcore_weak_ptr pvkernel)

# Function traits
add_executable(Tcore_function_traits function_traits.cpp)
target_link_libraries(Tcore_function_traits pvkernel)

# Const forward
add_executable(Tcore_const_fwd const_fwd.cpp)
target_link_libraries(Tcore_const_fwd pvkernel)

# Bit count
add_executable(Tcore_bit_count bit_count.cpp)
target_link_libraries(Tcore_bit_count pvkernel ${QT_LIBRARIES} ${OpenMP_LIBRARIES})

# Bit visitor
add_executable(Tcore_visit_bits visit_bits.cpp)
target_link_libraries(Tcore_visit_bits pvkernel ${QT_LIBRARIES} ${OpenMP_LIBRARIES})

# POD tree
add_executable(Tcore_podtree pod_tree.cpp)
target_link_libraries(Tcore_podtree pvkernel ${QT_LIBRARIES})

# Hardware concurrency
add_executable(Tcore_hardware_concurrency hardware_concurrency.cpp)
target_link_libraries(Tcore_hardware_concurrency pvkernel ${QT_LIBRARIES})

# Shared ptr forward
add_executable(Tcore_shared_ptr_fwd shared_ptr_fwd.cpp)
target_link_libraries(Tcore_shared_ptr_fwd pvkernel ${QT_LIBRARIES} ${OpenMP_LIBRARIES})

# Raid0
add_executable(Tcore_raid_test raid_test.cpp)
target_link_libraries(Tcore_raid_test pvkernel)

# Decimal storage
add_executable(Tcore_decimal_storage decimal_storage.cpp)
target_link_libraries(Tcore_decimal_storage pvkernel)

# Huge POD vector
add_executable(Tcore_huge_vector huge_vector.cpp)
target_link_libraries(Tcore_huge_vector pvkernel)

# lambda_connect
add_executable(Tcore_lambda_connect lambda_connect.cpp)
target_link_libraries(Tcore_lambda_connect pvkernel ${QT_LIBRARIES})
