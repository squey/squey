#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

include_directories(AFTER ${PVKERNEL_INCLUDE_DIR})
include_directories(AFTER ${PVHIVE_INCLUDE_DIR})
include_directories(AFTER ${PVGUIQT_INCLUDE_DIR})

set(COMMON_LIBRARIES ${PVGUIQT_LIBRARIES} ${PVHIVE_LIBRARIES} ${PVKERNEL_LIBRARIES} ${PICVIZ_LIBRARIES} ${TBB_LIBRARIES} ${BOOST_LIBRARIES} ${Boost_LIBRARIES_PYTHON} ${PYTHON_LIBRARY} ${OpenMP_LIBRARIES})

# Datatree model
add_executable(Tqt_datatree datatree-model.cpp)
target_link_libraries(Tqt_datatree ${COMMON_LIBRARIES})

# Picviz datatree model
add_executable(Tqt_picviz_tree picviz_tree.cpp common.cpp)
target_link_libraries(Tqt_picviz_tree ${COMMON_LIBRARIES})

# Axes combination model
qt4_wrap_cpp(AXES_COMB_MOC_SRC axes-comb_dlg.h OPTIONS ${QT4_WRAP_OPTIONS})
add_executable(Tqt_axes_comb axes-comb.cpp ${AXES_COMB_MOC_SRC})
target_link_libraries(Tqt_axes_comb ${COMMON_LIBRARIES})

# Picviz listing model
add_executable(Tqt_listing_model listing_model.cpp common.cpp)
target_link_libraries(Tqt_listing_model ${COMMON_LIBRARIES})
