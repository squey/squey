#
# Axis computation
#
include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libpicviz/src/include)

link_directories(${picviz-inspector_SOURCE_DIR}/libpicviz/src/)

set(FILTERS_DIST_LIST
)

set(FILTERS_DIST_FILES
)

set(LINK_LIBRARIES ${PICVIZ_LIBRARY} ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})

if(PYTHONLIBS_FOUND)
	# AG: BEFORE here is important, or boost does not include the good patchlevel.h (from Python), and does not
	# find the good Python version !!
	include_directories(BEFORE ${PYTHON_INCLUDE_DIRS})
	#add_library(axis_computation_python SHARED python/PVAxisComputationPython.cpp python/register.cpp)
	#target_link_libraries(axis_computation_python ${LINK_LIBRARIES} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES_PYTHON})
	set(FILTERS_DIST_LIST
	${FILTERS_DIST_LIST}
	axis_computation_python
	)
endif(PYTHONLIBS_FOUND)

INSTALL(TARGETS ${FILTERS_DIST_LIST} LIBRARY DESTINATION axis-computation
                                     RUNTIME DESTINATION axis-computation)
INSTALL(FILES ${FILTERS_DIST_FILES} DESTINATION axis-computation)
