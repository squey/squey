//! \file Window.cpp
//! Functions that handle window operations
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011

#ifdef USE_WTK_QT

#include <QCoreApplication>
#include <QApplication>

#include <GL/freeglut.h>

#include "../core/include/Window.h"
#include "include/global.h"
#include "include/PVGLWidget.h"

int PVGL::wtk_window_int_create(const char *name, int width, int height)
{
	return PVGL::WTKQt::Global::create_window(name, width, height);
}

void PVGL::wtk_window_resize(int width, int height)
{
	PVGL::WTKQt::Global::get_current_window()->resize(width, height);
}

void PVGL::wtk_window_fullscreen()
{
	QWidget* main_window = WTKQt::Global::get_current_window()->window();
	assert(main_window);
	main_window->showFullScreen();
}

void PVGL::wtk_window_need_redisplay()
{
	PVGL::WTKQt::Global::get_current_window()->update();
}

int PVGL::wtk_get_current_window()
{
	return WTKQt::Global::get_current_window_id();
}

void PVGL::wtk_set_current_window(int id)
{
	WTKQt::Global::set_current_window_id(id);
}

void PVGL::wtk_destroy_window(int id)
{
	QWidget* main_window = WTKQt::Global::get_widget_from_id(id)->window();
	assert(main_window);
	main_window->close();
	// Force deletion after closing
	main_window->deleteLater();
}

int PVGL::wtk_get_keyboard_modifiers()
{
	QCoreApplication* core_app = QCoreApplication::instance();
#ifndef NDEBUG
	QApplication* app = dynamic_cast<QApplication*>(core_app);
	assert(app);
#else
	QApplication* app = static_cast<QApplication*>(core_app);
#endif

	Qt::KeyboardModifiers modifiers = app->keyboardModifiers();

	// Translate this to GLUT's values
	int ret = 0;
	if ((modifiers & Qt::ShiftModifier) == Qt::ShiftModifier) {
		ret |= GLUT_ACTIVE_SHIFT;
	}
	if ((modifiers & Qt::ControlModifier) == Qt::ControlModifier) {
		ret |= GLUT_ACTIVE_CTRL;
	}
	if ((modifiers & Qt::AltModifier) == Qt::AltModifier) {
		ret |= GLUT_ACTIVE_ALT;
	}
	return ret;
}

#endif	// USE_WTK_QT
