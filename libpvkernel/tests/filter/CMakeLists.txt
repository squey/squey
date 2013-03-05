#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

set(LINK_LIBRARIES pvkernel ${QT_LIBRARIES} ${TBB_LIBRARIES} ${ICU_LIBRARY})

#add_executable(Tfilter_sharedptr shared_ptr.cpp)
#target_link_libraries(Tfilter_sharedptr ${LINK_LIBRARIES})
#add_test(Tfilter_sharedptr ${LINK_LIBRARIES})

declare_pv_test(Tfilter_filter_library filter_library.cpp)
target_link_libraries(Tfilter_filter_library ${LINK_LIBRARIES})
add_test(TestFilterFilterLibrary Tfilter_filter_library)
