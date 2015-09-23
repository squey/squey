#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

################################################
# Ressource declaration
################################################

set(LIBDISPLAYS_SRC_FILES
PVDisplayIf.cpp
PVDisplaysImpl.cpp
)

SET(LIBDISPLAYS_MOC_HEADERS
include/pvdisplays/PVDisplaysContainer.h
)

################################################
# Generate extra QT files (moc, uic, rcc)
################################################

set(SRC_FILES ${LIBDISPLAYS_SRC_FILES})
set(MOC_HEADERS ${LIBDISPLAYS_MOC_HEADERS})
QT5_WRAP_CPP(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})

################################################
# Declare the new library and linkage options
################################################

add_library(pvdisplays SHARED ${SRC_FILES} ${MOC_SRC})

set_target_properties(pvdisplays PROPERTIES SOVERSION 1)

set(QT5_LIBS Qt5Core Qt5Xml Qt5Widgets)

set(PVDISPLAYS_LINK_LIBRARIES picviz pvkernel ${QT5_LIBS} ${TBB_LIBRARIES} ${OpenMP_LIBRARIES})

target_link_libraries(pvdisplays ${PVDISPLAYS_LINK_LIBRARIES})

INSTALL(TARGETS pvdisplays LIBRARY DESTINATION .
                           RUNTIME DESTINATION .)
