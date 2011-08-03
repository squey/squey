#find_package(Boost)
#${Boost_LIBRARIES}

include_directories(AFTER ${PVCORE_INCLUDE_DIRS})
include_directories(AFTER ${PVRUSH_INCLUDE_DIRS})

#if (Boost_FOUND)
#	include_directories(AFTER ${Boost_INCLUDE_DIRS} )

add_executable(log2csv log2csv.cpp)
target_link_libraries(log2csv ${PVCORE_LIBRARY} ${PVFILTER_LIBRARY} ${PVRUSH_LIBRARY} ${QT_LIBRARIES} ${Boost_LIBRARIES})

add_executable(pvformatcheck pvformatcheck.cpp)
target_link_libraries(pvformatcheck ${PVCORE_LIBRARY} ${PVFILTER_LIBRARY} ${PVRUSH_LIBRARY} ${QT_LIBRARIES} ${Boost_LIBRARIES})

