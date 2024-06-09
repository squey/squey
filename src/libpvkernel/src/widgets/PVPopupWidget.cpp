//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/widgets/PVPopupWidget.h>
#include <qcoreevent.h>
#include <qguiapplication.h>
#include <qnamespace.h>
#include <qpoint.h>
#include <qrect.h>
#include <QApplication>
#include <QMouseEvent>
#include <QScreen>

static QRect fitToScreen(const QRect& rect)
{
	QRect screen_geom = QGuiApplication::screenAt(rect.center())->geometry();

	QRect new_geom = rect;

	if (rect.left() < screen_geom.left()) {
		new_geom.moveLeft(screen_geom.left());
	} else if (rect.right() > screen_geom.right()) {
		new_geom.moveLeft(screen_geom.right() - rect.width());
	}

	if (rect.top() < screen_geom.top()) {
		new_geom.moveTop(screen_geom.top());
	} else if (rect.bottom() > screen_geom.bottom()) {
		new_geom.moveTop(screen_geom.bottom() - rect.height());
	}

	return new_geom;
}

/*****************************************************************************
 * PVWidgets::PVPopupWidget::PVPopupWidget
 *****************************************************************************/

PVWidgets::PVPopupWidget::PVPopupWidget(QWidget* parent) : QWidget(parent)
{
	setFocusPolicy(Qt::ClickFocus);
	setMouseTracking(true);
}

/*****************************************************************************
 * PVWidgets::PVPopupWidget::popup
 *****************************************************************************/

void PVWidgets::PVPopupWidget::popup(const QPoint& p, bool centered)
{
	if (isVisible()) {
		return;
	}

	// make sure the popup's geometry is correct
	adjustSize();

	QPoint pos = p;
	QRect rect = geometry();

	if (centered) {
		pos -= QPoint(rect.width() / 2, rect.height() / 2);
	}

	rect.moveTopLeft(pos);
	rect = fitToScreen(rect);

	setGeometry(rect);
	raise();
	show();
}

/*****************************************************************************
 * PVWidgets::PVPopupWidget::setVisible
 *****************************************************************************/

void PVWidgets::PVPopupWidget::setVisible(bool visible)
{
	QWidget::setVisible(visible);

	if (visible) {
		setFocus();
	} else {
		parentWidget()->setFocus();
	}
}

/*****************************************************************************
 * PVWidgets::PVPopupWidget::is_close_key
 *****************************************************************************/

bool PVWidgets::PVPopupWidget::is_close_key(int key)
{
	return (key == Qt::Key_Escape);
}

/*****************************************************************************
 * PVWidgets::PVPopupWidget::mouseMoveEvent
 *****************************************************************************/

void PVWidgets::PVPopupWidget::mouseMoveEvent(QMouseEvent* event)
{
	QMouseEvent pevent(QEvent::MouseMove, parentWidget()->mapFromGlobal(event->globalPosition()),
	                   event->globalPosition(), event->button(), event->buttons(), event->modifiers());

	QApplication::sendEvent(parentWidget(), &pevent);
	event->ignore();
	QWidget::mouseMoveEvent(event);
}
