include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libpicviz/src/include)
add_definitions(${APR_FLAGS})
include_directories(AFTER ${APR_INCLUDES})
include_directories(AFTER ${PCAP_INCLUDE_DIRS})
include_directories(AFTER ${DUMBNET_INCLUDE_DIRS})

# if(WIN32)
# 	set(PICVIZ_LIBRARY "..\\src\\Debug\\picviz.lib")
# else(WIN32)
#   add_library(libpicviz STATIC IMPORTED)
#   set_property(TARGET libpicviz PROPERTY IMPORTED_LOCATION ${libpicviz_SOURCE_DIR}/src/libpicviz.a)
#   set(PICVIZ_LIBRARY libpicviz m)
# endif(WIN32)
if(WIN32)
	set(PICVIZ_LIBRARY picviz)
else(WIN32)
	set(PICVIZ_LIBRARY picviz m)
endif(WIN32)

set(LINK_LIBRARIES ${APR_LIBS} ${PCRE_PCRE_LIBRARY} ${PCAP_LIBRARIES} ${DUMBNET_LIBRARIES} ${PICVIZ_LIBRARY} ${PVCORE_LIBARY} ${PVRUSH_LIBRARY} ${QT_QTCORE_LIBRARY})

# add_executable(main main.c)

#add_executable(Tpicviz_rush2nraw pvrush_nraw.cpp)
#target_link_libraries(Tpicviz_rush2nraw ${LINK_LIBRARIES})


#add_executable(test_array array.c)
#target_link_libraries(test_array ${LINK_LIBRARIES})

add_executable(Tpicviz_color color.cpp)
target_link_libraries(Tpicviz_color ${LINK_LIBRARIES})

# add_executable(test_format format.c)
# target_link_libraries(test_format ${LINK_LIBRARIES})

# add_executable(init init.c)
# target_link_libraries(init ${LINK_LIBRARIES})

# add_executable(layer-stack layer-stack.c)
# target_link_libraries(layer-stack ${LINK_LIBRARIES})

# add_executable(line-properties line-properties.c)
# target_link_libraries(line-properties ${LINK_LIBRARIES})

#add_executable(test_lowlevel-api lowlevel-api.c)
#target_link_libraries(test_lowlevel-api ${LINK_LIBRARIES})

#add_executable(test_lowlevel-csv lowlevel-csv.c)
#target_link_libraries(test_lowlevel-csv ${LINK_LIBRARIES})

# add_executable(mapping mapping.c)
# target_link_libraries(mapping ${LINK_LIBRARIES})

# add_executable(pcap2view pcap-to-view.c)
# target_link_libraries(pcap2view ${LINK_LIBRARIES})

add_executable(Tpicviz_selection selection.cpp)
target_link_libraries(Tpicviz_selection ${LINK_LIBRARIES})

# add_executable(type-discovery type-discovery.c)
# target_link_libraries(type-discovery ${LINK_LIBRARIES})

# add_executable(variables variables.c)
# target_link_libraries(variables ${LINK_LIBRARIES})

# add_executable(snortmypcap snortmypcap.c)
# target_link_libraries(snortmypcap ${LINK_LIBRARIES})

# add_executable(diffsyslog diffsyslog.c)
# target_link_libraries(diffsyslog ${LINK_LIBRARIES})

# add_executable(filetype filetype-discover.c)
# target_link_libraries(filetype ${LINK_LIBRARIES})

#add_executable(testpythonctype python-ctype.c)
#target_link_libraries(mapping ${LINK_LIBRARIES} python2.6)

#add_executable(test_arguments arguments.c)
#target_link_libraries(test_arguments ${LINK_LIBRARIES})

#add_executable(test_strptime strptime.c)
#target_link_libraries(test_strptime ${LINK_LIBRARIES})

#add_executable(test_filtersplugins filters-plugins.c)
#target_link_libraries(test_filtersplugins ${LINK_LIBRARIES})

# add_executable(test_readpcap read-pcap.c)
# target_link_libraries(test_readpcap ${LINK_LIBRARIES})

#add_executable(test_pluginsdir pluginsdir.c)
#target_link_libraries(test_pluginsdir ${LINK_LIBRARIES})

#add_executable(test_functions functions.c)
#target_link_libraries(test_functions ${LINK_LIBRARIES})

#add_executable(test_view view.c)
#target_link_libraries(test_view ${LINK_LIBRARIES})

#add_executable(test_axescombination axes-combination.c)
#target_link_libraries(test_axescombination ${LINK_LIBRARIES})

#add_executable(test_readfile readfile.c)
#target_link_libraries(test_readfile ${LINK_LIBRARIES})

# add_executable(test_format-stack format-stack.c)
# target_link_libraries(test_format-stack ${LINK_LIBRARIES})

#add_executable(test_open-save open-save.c)
#target_link_libraries(test_open-save ${LINK_LIBRARIES})

#add_executable(test_bla bla.c)
#target_link_libraries(test_bla ${LINK_LIBRARIES})

#add_executable(test_file file.c)
#target_link_libraries(test_file ${LINK_LIBRARIES})

#add_executable(test_string string.c)
#target_link_libraries(test_string ${LINK_LIBRARIES})

#add_executable(test_source source.c)
#target_link_libraries(test_source ${LINK_LIBRARIES})

enable_testing()

#add_test(TestRushNraw Tpicviz_rush2nraw)

# add_test(TestMain main)
# add_test(TestInit init)
# add_test(TestPcap2View pcap2view)
# add_test(TestCSV2View csv2view)
add_test(TestColor test_color)
add_test(TestSelection test_selection)
# add_test(TestTypeDiscovery type-discovery)
# add_test(TestLayerStack layer-stack)
# add_test(TestLineProperties line-properties)
#add_test(TestLowLevelAPI test_lowlevel-api)
# add_test(TestNormalizePGDL normalize-pgdl)
# add_test(TestVariables variables)
# add_test(TestFormat test_format)
# add_test(TestLowLevelCSV test_lowlevel-csv)
# add_test(TestMapping mapping)
# add_test(TestArray array)
# add_test(TestSnortMyPCAP snortmypcap)
# add_test(TestDiffSyslog diffsyslog)
# add_test(TestFileType filetype)
#add_test(TestPythonCtype testpythonctype)
#add_test(TestArguments test_arguments)
#add_test(TestStrptime test_strptime)
#add_test(TestFiltersPlugins test_filtersplugins)
# add_test(TestReadPCAP test_readpcap)
#add_test(TestPluginsDir test_pluginsdir)
#add_test(TestFunctions test_functions)
#add_test(TestView test_view)
#add_test(TestAxesCombination test_axescombination)
#add_test(TestReadFile test_readfile)
# add_test(TestFormatStack test_format-stack)
#add_test(TestOpenSave test_open-save)
#add_test(TestBigLogArray test_bla)
#add_test(TestFile test_file)
#add_test(TestString test_string)
#add_test(TestSource test_source)

