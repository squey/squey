#
# @file
#

project(pcap-gui-widget)

find_package(Qt5Core REQUIRED)
include_directories(${Qt5Core_INCLUDE_DIRS})

find_package(Qt5Widgets REQUIRED)
include_directories(${Qt5Widgets_INCLUDE_DIRS})

find_package(Qt5Gui REQUIRED)
include_directories(${Qt5Gui_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Core_EXECUTABLE_COMPILE_FLAGS}")

include_directories("${CMAKE_SOURCE_DIR}/libpvpcap/include/")
#include_directories(AFTER ${pcap-gui-widget_SOURCE_DIR}/src)

# Add the binary dir as ui file we be generated there
include_directories(AFTER ${pcap-gui-widget_BINARY_DIR}/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

#SET(PCAP_GUI_QRC resources.qrc )

SET(PCAP_GUI_MOC_HEADER
	import-ui/src/include/ProgressWidget.h
    import-ui/src/include/SelectionWidget.h

    profile-ui/src/include/EditionWidget.h
    profile-ui/src/include/OverviewWidget.h
    profile-ui/src/include/ProfileWidget.h
    profile-ui/src/include/TreeWidget.h
    profile-ui/src/include/OptionWidget.h

    profile-ui/models/include/PcapTreeModel.h
    profile-ui/models/include/PcapTreeSelectionModel.h
    profile-ui/models/include/ProtocolFieldListModel.h
)

SET(PCAP_GUI_SRC
	pcap-gui.cpp

	import-ui/src/ProgressWidget.cpp
    import-ui/src/SelectionWidget.cpp

    profile-ui/src/EditionWidget.cpp
    profile-ui/src/OverviewWidget.cpp
    profile-ui/src/ProfileWidget.cpp
    profile-ui/src/TreeWidget.cpp
    profile-ui/src/OptionWidget.cpp

    profile-ui/models/PcapTreeModel.cpp
    profile-ui/models/PcapTreeSelectionModel.cpp
    profile-ui/models/ProtocolFieldListModel.cpp

)

SET(PCAP_GUI_UI
	import-ui/forms/ProgressWidget.ui
    import-ui/forms/SelectionWidget.ui

    profile-ui/forms/EditionWidget.ui
    profile-ui/forms/OverviewWidget.ui
    profile-ui/forms/ProfileWidget.ui
    profile-ui/forms/TreeWidget.ui
    profile-ui/forms/OptionWidget.ui
)

QT5_WRAP_UI(UI_SRC ${PCAP_GUI_UI})
QT5_WRAP_CPP(MOC_SRC ${PCAP_GUI_MOC_HEADER} OPTIONS ${QT5_WRAP_OPTIONS})
#QT5_ADD_RESOURCES(QRC_FILES_SRCS ${PCAP_GUI_QRC})

#add_executable(pcapsicum-gui ${PCAP_GUI_SRC} ${MOC_SRC} ${QRC_FILES_SRCS})
#target_link_libraries(pcapsicum-gui Qt5Widgets Qt5Gui Qt5Core pcapsicum)

# Set rpath
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${PVKERNEL_PLUGIN}/input-types/")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_library(pcap-gui-widget SHARED ${PCAP_GUI_SRC} ${UI_SRC} ${MOC_SRC}) # ${QRC_FILES_SRCS})
target_link_libraries(pcap-gui-widget Qt5Widgets Qt5Gui Qt5Core pvpcap)

#INSTALL(TARGETS pcapsicum-gui DESTINATION bin)
#INSTALL(FILES ${CMAKE_SOURCE_DIR}/pcapsicum-gui/config.ini DESTINATION bin)

set(INPUTS_DIST_LIST
	pcap-gui-widget
	)

INSTALL(TARGETS ${INPUTS_DIST_LIST} LIBRARY DESTINATION ${PVKERNEL_PLUGIN}/input-types
				    RUNTIME DESTINATION ${PVKERNEL_PLUGIN}/input-types)
