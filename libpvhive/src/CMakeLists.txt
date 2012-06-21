
include_directories(AFTER ${ICU_INCLUDE})

set(PVHIVE_SRC_FILES
PVHive.cpp
PVActorBase.cpp
PVObserver.cpp
)

SET(PVHIVE_MOC_HEADERS
include/pvhive/PVRefreshSignal.h
include/pvhive/PVQRefresh.h
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

set(SRC_FILES ${PVHIVE_SRC_FILES})
set(HEADER_FILES ${PVHIVE_HEADER_FILES})
set(MOC_HEADERS ${PVHIVE_MOC_HEADERS})
qt4_wrap_cpp(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})

add_library(pvhive SHARED ${SRC_FILES} ${HEADER_FILES} ${MOC_SRC} ${MOC_HEADERS})

set_target_properties(pvhive PROPERTIES SOVERSION 1)

set(PVHIVE_LINK_LIBRARIES ${QT_LIBRARIES} ${Boost_LIBRARIES_PYTHON})

target_link_libraries(pvhive ${PVHIVE_LINK_LIBRARIES})

INSTALL(TARGETS pvhive LIBRARY DESTINATION .
		       RUNTIME DESTINATION .)
