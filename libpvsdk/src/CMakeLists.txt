include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libpvsdk/src/include)
include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libpicviz/src/include)

# Qt
include(${QT_USE_FILE})

SET(SRC_FILES
PVMessenger.cpp
)

SET(HEADER_FILES
include/pvsdk/PVMessenger.h
)

add_library(pvsdk SHARED ${SRC_FILES} ${HEADER_FILES})
set_target_properties(pvsdk PROPERTIES SOVERSION 1)

target_link_libraries(pvsdk ${PICVIZ_LIBRARY} ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY} ${PVKERNEL_LIBRARIES} ${TBB_LIBRARIES} ${Boost_LIBRARIES})

INSTALL(TARGETS pvsdk DESTINATION .)
