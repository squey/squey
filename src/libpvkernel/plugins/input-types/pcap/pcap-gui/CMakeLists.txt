#
# @file
#

project(pcap-gui-widget)

set(PCAP_GUI_MOC_HEADERS
    CapturesPanel.h
    ColumnsPanel.h
    FieldsPanel.h
    FilterPanel.h
    ImportDialog.h
    OptionsPanel.h
    PreviewPanel.h
    ProfileEditor.h
    ProfilesPanel.h
)

set(PCAP_GUI_SRC
    CapturesPanel.cpp
    ColumnsPanel.cpp
    FieldsPanel.cpp
    FilterPanel.cpp
    ImportDialog.cpp
    OptionsPanel.cpp
    PreviewPanel.cpp
    ProfileEditor.cpp
    ProfilesPanel.cpp
    Style.cpp
)

qt6_wrap_cpp(MOC_SRC ${PCAP_GUI_MOC_HEADERS} OPTIONS)

add_library(pcap-gui-widget STATIC ${PCAP_GUI_SRC} ${MOC_SRC})
target_include_directories(pcap-gui-widget PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(pcap-gui-widget PRIVATE Qt6::Widgets Qt6::Gui Qt6::Core pvpcap)
