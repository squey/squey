#
# @file
#

add_squey_compile_test(Twidgets_color-picker LINKS ${LINK_LIBRARIES} FILES color-picker.cpp)
add_squey_compile_test(Twidgets_color-dialog LINKS ${LINK_LIBRARIES} FILES color-dialog.cpp)

QT6_WRAP_CPP(PVGV_MOC mygraphicsview.h OPTIONS ${QT6_WRAP_OPTIONS})
add_squey_compile_test(Epvgraphicsview LINKS pvkernel FILES pvgraphicsview.cpp ${PVGV_MOC})
add_squey_compile_test(Epvgraphicsview_inv_y LINKS pvkernel FILES graphicsview_inv_y.cpp)

file(RELATIVE_PATH REL_TEST_DIR ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake DESTINATION "tests/${REL_TEST_DIR}" COMPONENT tests)