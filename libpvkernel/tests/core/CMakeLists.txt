include_directories(AFTER ${ICU_INCLUDE})

# if(WIN32)
# 	set(PVCORE_LIBRARY pvcore)
# else(WIN32)
# 	set(PVCORE_LIBRARY pvcore m)
# endif(WIN32)

add_executable(Tcore_capslock capslock.cpp)
target_link_libraries(Tcore_capslock ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})
add_test(TestCapslock Tcore_capslock)

add_executable(Tcore_simple simple.cpp)
target_link_libraries(Tcore_simple ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})
add_test(TestSimple Tcore_simple)

#add_executable(Tcore_debug debug.cpp)
#target_link_libraries(Tcore_debug ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})
#add_test(TestDebug Tcore_debug)

add_executable(Tcore_network network.cpp)
target_link_libraries(Tcore_network ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})
add_test(TestNetwork Tcore_network)

# add_executable(Tcore_process process.cpp)
# target_link_libraries(Tcore_process ${PVLIBRARIES} pthread)
# add_test(TestProcess Tcore_process)

add_executable(Tcore_sysinfo sysinfo.cpp)
target_link_libraries(Tcore_sysinfo ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})
add_test(TestSysinfo Tcore_sysinfo)

add_executable(Tcore_logger logger.cpp)
target_link_libraries(Tcore_logger ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})
add_test(TestLogger Tcore_logger)

# Time format parser test
add_executable(Tcore_timeformat time-format.cpp)
target_link_libraries(Tcore_timeformat ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES} ${ICU_LIBRARY})
add_test(TestTimeformat Tcore_timeformat test-files/time_format)

# Time format performance
add_executable(Tcore_timeformat_perf time-format-perf.cpp)
target_link_libraries(Tcore_timeformat_perf ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES} ${ICU_LIBRARY})

# Serialization
add_executable(Tcore_serialize serialize.cpp)
target_link_libraries(Tcore_serialize ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})

# Matrix
add_executable(Tcore_matrix matrix.cpp)
target_link_libraries(Tcore_matrix ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})
