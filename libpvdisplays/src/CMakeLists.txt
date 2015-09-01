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

set(LIBDISPLAYS_HEADER_FILES
include/pvdisplays/PVDisplayIf.h
include/pvdisplays/PVDisplaysImpl.h
)

SET(LIBDISPLAYS_MOC_HEADERS
include/pvdisplays/PVDisplaysContainer.h
)

SET(UI_FILES
)

SET(QRC_FILES
)

################################################
# Generate extra QT files (moc, uic, rcc)
################################################

set(SRC_FILES ${LIBDISPLAYS_SRC_FILES})
set(HEADER_FILES ${LIBDISPLAYS_HEADER_FILES})
set(MOC_HEADERS ${LIBDISPLAYS_MOC_HEADERS})
QT5_WRAP_CPP(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})
QT5_ADD_RESOURCES(QRC_FILES_SRCS ${QRC_FILES})
QT5_WRAP_UI(UI_SRC ${UI_FILES})

################################################
# Declare the new library and linkage options
################################################

add_library(pvdisplays SHARED ${SRC_FILES} ${HEADER_FILES} ${MOC_SRC} ${MOC_HEADERS} ${QRC_FILES_SRCS} ${UI_FILES})

set_target_properties(pvdisplays PROPERTIES SOVERSION 1)

set(QT5_LIBS Qt5Core Qt5Xml Qt5Widgets)

set(PVDISPLAYS_LINK_LIBRARIES picviz pvkernel ${QT5_LIBS} ${TBB_LIBRARIES} ${OpenMP_LIBRARIES})

target_link_libraries(pvdisplays ${PVDISPLAYS_LINK_LIBRARIES})

INSTALL(TARGETS pvdisplays LIBRARY DESTINATION .
                           RUNTIME DESTINATION .)
