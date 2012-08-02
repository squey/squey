/**
 * \file PVGLWidget.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifdef USE_WTK_QT

#include "include/PVGLWidget.h"
#include "include/global.h"

PVGL::WTKQt::PVGLWidget::PVGLWidget(int id, QWidget* parent):
	QGLWidget(parent),
	_win_id(id)
{
}

void PVGL::WTKQt::PVGLWidget::initializeGL()
{
}

void PVGL::WTKQt::PVGLWidget::resizeGL(int w, int h)
{
	if (!Global::_callback_reshape) {
		return;
	}

	assert(Global::is_current_window(this));
	Global::_callback_reshape(w, h);
}

void PVGL::WTKQt::PVGLWidget::paintGL()
{
	if (!Global::_callback_idle) {
		return;
	}

	assert(Global::is_current_window(this));
	Global::_callback_idle();
}

#endif
