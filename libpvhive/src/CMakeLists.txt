#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

include_directories(AFTER ${ICU_INCLUDE})

################################################
# Ressource declaration
################################################

set(PVHIVE_SRC_FILES
PVHive.cpp
PVActorBase.cpp
PVObserver.cpp
PVRefreshSignal.cpp
PVFuncObserver.cpp
)

set(PVHIVE_WAXES_FILES
waxes/inendi/PVView.cpp
)

SET(PVHIVE_MOC_HEADERS
include/pvhive/PVRefreshSignal.h
include/pvhive/PVQRefresh.h
include/pvhive/PVFuncObserver.h
)

################################################
# Generate extra QT files (moc, uic, rcc)
################################################

set(SRC_FILES ${PVHIVE_SRC_FILES} ${PVHIVE_WAXES_FILES})
set(MOC_HEADERS ${PVHIVE_MOC_HEADERS})
qt5_wrap_cpp(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})

################################################
# Declare the new library and linkage options
################################################

add_library(pvhive SHARED ${SRC_FILES} ${MOC_SRC})

set_target_properties(pvhive PROPERTIES SOVERSION 1)

set(QT5_LIBS Qt5Core Qt5Xml Qt5Gui)

set(PVHIVE_LINK_LIBRARIES ${QT5_LIBS} ${TBB_LIBRARIES} inendi pvkernel)

target_link_libraries(pvhive ${PVHIVE_LINK_LIBRARIES})

INSTALL(TARGETS pvhive LIBRARY DESTINATION lib
		       RUNTIME DESTINATION lib)
