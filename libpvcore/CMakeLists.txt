#set(QT_DONT_USE_QTGUI true)
# QT XML support
set(QT_USE_QTXML TRUE)
include(${QT_USE_FILE})


if(WIN32)
        set(PVCORE_SHARE_DIR "share")
else(WIN32)
        set(PVCORE_SHARE_DIR "${CMAKE_INSTALL_PREFIX}/share/pvcore/")
endif(WIN32)
add_definitions(-DPVCORE_SHARE_DIR="${PVCORE_SHARE_DIR}")

add_subdirectory(src)
add_subdirectory(tests)
