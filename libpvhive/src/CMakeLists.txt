#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

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
waxes/picviz/PVScene.cpp
waxes/picviz/PVMapped.cpp
waxes/picviz/PVSource.cpp
waxes/picviz/PVView.cpp
waxes/picviz/PVRoot.cpp
waxes/picviz/PVPlotted.cpp
)

if(ENABLE_CORRELATION)
	list(APPEND PVHIVE_WAXES_FILES
	waxes/picviz/PVAD2GView.cpp
	)
endif()

SET(PVHIVE_MOC_HEADERS
include/pvhive/PVRefreshSignal.h
include/pvhive/PVQRefresh.h
include/pvhive/PVFuncObserver.h
)

set(PVHIVE_HEADER_FILES
include/pvhive/PVHive.h
include/pvhive/PVActor.h
include/pvhive/PVObserver.h
include/pvhive/PVObserverCallback.h
include/pvhive/PVObserverSignal.h
include/pvhive/PVRefreshSignal.h
include/pvhive/PVQObserver.h
include/pvhive/PVQRefresh.h
)

################################################
# Generate extra QT files (moc, uic, rcc)
################################################

set(SRC_FILES ${PVHIVE_SRC_FILES} ${PVHIVE_WAXES_FILES})
set(HEADER_FILES ${PVHIVE_HEADER_FILES})
set(MOC_HEADERS ${PVHIVE_MOC_HEADERS})
qt5_wrap_cpp(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})

################################################
# Declare the new library and linkage options
################################################

add_library(pvhive SHARED ${SRC_FILES} ${HEADER_FILES} ${MOC_SRC} ${MOC_HEADERS})

set_target_properties(pvhive PROPERTIES SOVERSION 1)

set(QT5_LIBS Qt5Core Qt5Xml Qt5Gui)

set(PVHIVE_LINK_LIBRARIES ${QT5_LIBS} ${TBB_LIBRARIES} picviz pvkernel)

target_link_libraries(pvhive ${PVHIVE_LINK_LIBRARIES})

INSTALL(TARGETS pvhive LIBRARY DESTINATION .
		       RUNTIME DESTINATION .)
