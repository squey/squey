#include "include/global.h"
#include "include/PVGLWidget.h"

void PVGL::WTKQt::Global::set_current_window(PVGLWidget* widget)
{
	_cur_win_id = widget->id();
	_cur_win = widget;
}

PVGL::WTKQt::PVGLWidget* PVGL::WTKQt::Global::get_widget_from_id(int id)
{
	assert(_windows.find(id) != _windows.end());
	return _windows[id];
}

void PVGL::WTKQt::Global::set_ms_start()
{
	_init_start = tbb::tick_count::now();
}
