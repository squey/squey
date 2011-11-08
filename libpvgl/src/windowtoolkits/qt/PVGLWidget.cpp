#include "include/PVGLWidget.h"


PVGL::WTLQt::PVGLWidget::PVGLWidget(int id, QWidget* parent = 0):
	QGLWidget(parent),
	_win_id(id)
{
}

void PVGL::WTLQt::PVGLWidget::initializeGL()
{
}

void PVGL::WTLQt::PVGLWidget::resizeGL(int w, int h)
{
	Global::set_current_window(this);
	Global::_callback_reshape(w, h);
}

void PVGL::WTLQt::PVGLWidget::paintGL()
{
	Global::set_current_window(this);
	Global::_callback_idle();
}
