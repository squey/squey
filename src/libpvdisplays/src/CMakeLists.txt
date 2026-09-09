#
# @file
#

################################################
# Ressource declaration
################################################

set(LIBDISPLAYS_SRC_FILES
PVDisplayIf.cpp
PVDisplaysImpl.cpp
)

set(LIBDISPLAYS_MOC_HEADERS
../include/pvdisplays/PVDisplaysContainer.h
)

################################################
# Generate extra QT files (moc, uic, rcc)
################################################

set(SRC_FILES ${LIBDISPLAYS_SRC_FILES})
set(MOC_HEADERS ${LIBDISPLAYS_MOC_HEADERS})
qt6_wrap_cpp(MOC_SRC ${MOC_HEADERS} OPTIONS)

################################################
# Declare the new library and linkage options
################################################

add_library(pvdisplays SHARED ${SRC_FILES} ${MOC_SRC})
set_target_properties(pvdisplays PROPERTIES SOVERSION 1)
squey_speedup_compilation(pvdisplays QT)

target_include_directories(pvdisplays PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)

target_link_libraries(pvdisplays
	PUBLIC pvsquey pvkernel Qt6::Core Qt6::Xml Qt6::Widgets
	PRIVATE ${TBB_LIBRARIES} ${OpenMP_LIBRARIES})

install(TARGETS pvdisplays LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
					RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR})
