include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpvcore/src/include)
include_directories(AFTER ${ICU_INCLUDE})

if(WIN32)
	set(PVCORE_LIBRARY pvcore)
else(WIN32)
	set(PVCORE_LIBRARY pvcore m)
endif(WIN32)

add_executable(Tcore_simple simple.cpp)
target_link_libraries(Tcore_simple ${PVCORE_LIBRARY} ${QT_LIBRARIES})
add_test(TestSimple Tcore_simple)

#add_executable(Tcore_debug debug.cpp)
#target_link_libraries(Tcore_debug ${PVCORE_LIBRARY} ${QT_LIBRARIES})
#add_test(TestDebug Tcore_debug)

add_executable(Tcore_network network.cpp)
target_link_libraries(Tcore_network ${PVCORE_LIBRARY} ${QT_LIBRARIES})
add_test(TestNetwork Tcore_network)

# add_executable(Tcore_process process.cpp)
# target_link_libraries(Tcore_process ${PVLIBRARIES} pthread)
# add_test(TestProcess Tcore_process)

add_executable(Tcore_sysinfo sysinfo.cpp)
target_link_libraries(Tcore_sysinfo ${PVCORE_LIBRARY} ${QT_LIBRARIES})
add_test(TestSysinfo Tcore_sysinfo)

add_executable(Tcore_logger logger.cpp)
target_link_libraries(Tcore_logger ${PVCORE_LIBRARY} ${QT_LIBRARIES})
add_test(TestLogger Tcore_logger)

# Time format parser test
add_executable(Tcore_timeformat time-format.cpp)
target_link_libraries(Tcore_timeformat ${PVCORE_LIBRARY} ${QT_LIBRARIES} ${ICU_LIBRARY})
add_test(TestTimeformat Tcore_timeformat test-files/time_format)

add_executable(Tcore_parser parser.cpp)
target_link_libraries(Tcore_parser ${PVCORE_LIBRARY} ${QT_LIBRARIES})
add_test(TestParser Tcore_parser)

