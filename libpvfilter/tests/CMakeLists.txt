include_directories(AFTER ${PVRUSH_INCLUDE_DIRS})
include_directories(AFTER ${PVFILTER_INCLUDE_DIRS})
include_directories(AFTER ${PVCORE_INCLUDE_DIRS})
include_directories(AFTER ${QT_QTCORE_INCLUDE_DIR})
include_directories(AFTER ${QT_INCLUDE_DIR})

if(WIN32)
	set(PVRUSH_LIBRARY pvrush)
else(WIN32)
	set(PVRUSH_LIBRARY pvrush m)
endif(WIN32)

set(PVFILTER_LIBRARY pvfilter)
set(LINK_LIBRARIES ${PVCORE_LIBRARY} ${PVFILTER_LIBRARY} ${QT_LIBRARIES} ${TBB_LIBRARIES} ${TBB_MALLOC_LIBRARY} ${ICU_LIBRARY})

add_executable(Tfilter_sharedptr shared_ptr.cpp)
target_link_libraries(Tfilter_sharedptr ${LINK_LIBRARIES})
add_test(Tfilter_sharedptr ${LINK_LIBRARIES})

