include_directories(AFTER ${QT_QTCORE_INCLUDE_DIR})
include_directories(AFTER ${QT_INCLUDE_DIR})

set(LINK_LIBRARIES ${PVKERNEL_LIBRARY} ${QT_LIBRARIES} ${TBB_LIBRARIES} ${TBB_MALLOC_LIBRARY} ${ICU_LIBRARY})

add_executable(Tfilter_sharedptr shared_ptr.cpp)
target_link_libraries(Tfilter_sharedptr ${LINK_LIBRARIES})
add_test(Tfilter_sharedptr ${LINK_LIBRARIES})

add_executable(Tfilter_filter_library filter_library.cpp)
target_link_libraries(Tfilter_filter_library ${LINK_LIBRARIES})
add_test(Tfilter_filter_library ${LINK_LIBRARIES})
