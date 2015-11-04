#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

add_pv_compile_test(Twidgets_datatreemodel LINKS pvkernel FILES datatree-model.cpp)
add_pv_compile_test(Twidgets_datatreeproxy LINKS pvkernel FILES datatree-proxy.cpp)
add_pv_compile_test(Twidgets_color-picker LINKS pvkernel FILES color-picker.cpp)
add_pv_compile_test(Twidgets_color-dialog LINKS pvkernel FILES color-dialog.cpp)

QT5_WRAP_CPP(PVGV_MOC mygraphicsview.h OPTIONS ${QT5_WRAP_OPTIONS})
add_pv_compile_test(Epvgraphicsview LINKS pvkernel FILES pvgraphicsview.cpp ${PVGV_MOC})
add_pv_compile_test(Epvgraphicsview_inv_y LINKS pvkernel FILES graphicsview_inv_y.cpp)
