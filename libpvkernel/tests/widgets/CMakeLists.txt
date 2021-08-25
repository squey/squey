#
# @file
#

add_inspector_compile_test(Twidgets_color-picker LINKS ${LINK_LIBRARIES} FILES color-picker.cpp)
add_inspector_compile_test(Twidgets_color-dialog LINKS ${LINK_LIBRARIES} FILES color-dialog.cpp)

QT5_WRAP_CPP(PVGV_MOC mygraphicsview.h OPTIONS ${QT5_WRAP_OPTIONS})
add_inspector_compile_test(Epvgraphicsview LINKS pvkernel FILES pvgraphicsview.cpp ${PVGV_MOC})
add_inspector_compile_test(Epvgraphicsview_inv_y LINKS pvkernel FILES graphicsview_inv_y.cpp)
