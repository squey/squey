include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpicviz/src/include)

add_definitions(${PICVIZ_DEFINITIONS})
include_directories(AFTER ${PICVIZ_INCLUDE_DIRS})


if(WIN32)
	set(PICVIZ_LIBRARY picviz)
else(WIN32)
	set(PICVIZ_LIBRARY picviz m)
endif(WIN32)

set(LINK_LIBRARIES ${APR_LIBS} ${PCRE_PCRE_LIBRARY} ${PCAP_LIBRARIES} ${DUMBNET_LIBRARIES} ${PICVIZ_LIBRARY} ${PVCORE_LIBRARY} ${PVRUSH_LIBRARY} ${QT_LIBRARIES} ${TBB_LIBRARY} -lrt)

enable_testing()

# add_executable(Tpicviz_color color.cpp)
# target_link_libraries(Tpicviz_color ${LINK_LIBRARIES})
# add_test(TestColor Tpicviz_color)

# add_executable(Tpicviz_time-plotting time-plotting.cpp)
# target_link_libraries(Tpicviz_time-plotting ${LINK_LIBRARIES})
# add_test(TestTimePlotting Tpicviz_time-plotting)

# add_executable(Tpicviz_float-plotting float-plotting.cpp)
# target_link_libraries(Tpicviz_float-plotting ${LINK_LIBRARIES})
# add_test(TestFloatPlotting Tpicviz_float-plotting)

add_executable(Tpicviz_root root.cpp)
target_link_libraries(Tpicviz_root ${LINK_LIBRARIES})
add_test(TestRoot Tpicviz_root)

# add_executable(Tpicviz_mapping-function mapping-function.cpp)
# target_link_libraries(Tpicviz_mapping-function ${LINK_LIBRARIES})
# add_test(TestMappingFunction Tpicviz_mapping-function)

#add_executable(Tpicviz_mapping mapping.cpp)
#target_link_libraries(Tpicviz_mapping ${LINK_LIBRARIES})
#add_test(TestMapping Tpicviz_mapping)

add_executable(Tpicviz_mapping_time mapping-time.cpp)
target_link_libraries(Tpicviz_mapping_time ${LINK_LIBRARIES})
add_test(TestMapping_time Tpicviz_mapping_time)

#add_executable(Tpicviz_plotting plotting.cpp)
#target_link_libraries(Tpicviz_plotting ${LINK_LIBRARIES})
#add_test(TestPlotting Tpicviz_plotting)



add_executable(TPVSelection PVSelection.cpp)
target_link_libraries(TPVSelection ${LINK_LIBRARIES})
add_test(TestPVSelection TPVSelection)


#add_executable(Tpicviz_source source.cpp)
#target_link_libraries(Tpicviz_source ${LINK_LIBRARIES})
#add_test(TestSource Tpicviz_source)

#add_executable(Tpicviz_filtersList filter_factory.cpp)
#target_link_libraries(Tpicviz_filtersList ${LINK_LIBRARIES})
#add_test(TestFiltersList Tpicviz_filtersList)
