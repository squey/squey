/**
 * \file global.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifdef USE_WTK_QT

#include "include/global.h"
#include "include/PVGLWidget.h"

#include <QDialog>
#include <QVBoxLayout>

int PVGL::WTKQt::Global::_cur_win_id = -1;
PVGL::WTKQt::PVGLWidget* PVGL::WTKQt::Global::_cur_win = NULL;

PVGL::pvgl_callback_close_t PVGL::WTKQt::Global::_callback_close = 0;
PVGL::pvgl_callback_display_t PVGL::WTKQt::Global::_callback_display = 0;
PVGL::pvgl_callback_entry_t PVGL::WTKQt::Global::_callback_entry = 0;
PVGL::pvgl_callback_idle_t PVGL::WTKQt::Global::_callback_idle = 0;
PVGL::pvgl_callback_keyboard_t PVGL::WTKQt::Global::_callback_keyboard = 0;
PVGL::pvgl_callback_motion_t PVGL::WTKQt::Global::_callback_motion = 0;
PVGL::pvgl_callback_passive_motion_t PVGL::WTKQt::Global::_callback_passive_motion = 0;
PVGL::pvgl_callback_mouse_t PVGL::WTKQt::Global::_callback_mouse = 0;
PVGL::pvgl_callback_reshape_t PVGL::WTKQt::Global::_callback_reshape = 0;
PVGL::pvgl_callback_special_t PVGL::WTKQt::Global::_callback_special = 0;
PVGL::WTKQt::map_glwindows PVGL::WTKQt::Global::_windows;
tbb::tick_count PVGL::WTKQt::Global::_init_start;

#if 0
void PVGL::WTKQt::Global::set_current_window(PVGLWidget* widget)
{
	_cur_win_id = widget->id();
	_cur_win = widget;
}
#endif

PVGL::WTKQt::PVGLWidget* PVGL::WTKQt::Global::get_widget_from_id(int id)
{
	assert(_windows.find(id) != _windows.end());
	return _windows[id];
}

void PVGL::WTKQt::Global::set_ms_start()
{
	_init_start = tbb::tick_count::now();
}

int PVGL::WTKQt::Global::create_window(const char* name, int width, int height)
{
	int new_id = _windows.size();
	QDialog* dlg = new QDialog();
	dlg->resize(width, height);
	dlg->setWindowTitle(name);

	PVGLWidget* gl_widget = new PVGLWidget(new_id);
	QVBoxLayout* layout = new QVBoxLayout();
	layout->addWidget(gl_widget);
	dlg->setLayout(layout);

	_windows.insert(map_glwindows::value_type(new_id, gl_widget));

	if (new_id == 0) {
		// That's the first one
		set_current_window_id(0);
	}

	dlg->show();

	return new_id;
}

void PVGL::WTKQt::Global::set_current_window_id(int id)
{
	assert(_windows.find(id) != _windows.end());
	PVGLWidget* new_widget = _windows[id];

	if (_cur_win) {
		_cur_win->window()->clearFocus();
	}

	_cur_win_id = id;
	_cur_win = new_widget;
	new_widget->window()->setFocus();
}

#endif
