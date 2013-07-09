
#include <pvkernel/widgets/PVPopupWidget.h>

#include <QApplication>
#include <QDesktopWidget>
#include <QMouseEvent>

static QRect fitToScreen(const QRect &rect)
{
	const QDesktopWidget *dw = QApplication::desktop();
	QRect screen_geom = dw->availableGeometry(dw->screenNumber(rect.center()));

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

PVWidgets::PVPopupWidget::PVPopupWidget(QWidget* parent) :
	QWidget(parent)
{
	setFocusPolicy(Qt::ClickFocus);
	setMouseTracking(true);
}

/*****************************************************************************
 * PVWidgets::PVPopupWidget::popup
 *****************************************************************************/

void PVWidgets::PVPopupWidget::popup(const QPoint& p, bool centered)
{
	if(isVisible()) {
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
 * PVWidgets::PVPopupWidget::mouseMoveEvent
 *****************************************************************************/

void PVWidgets::PVPopupWidget::mouseMoveEvent(QMouseEvent* event)
{
	QMouseEvent pevent(QEvent::MouseMove,
	                   parentWidget()->mapFromGlobal(event->globalPos()),
	                   event->globalPos(),
	                   event->button(),
	                   event->buttons(),
	                   event->modifiers());

	QApplication::sendEvent(parentWidget(), &pevent);
	event->ignore();
	QWidget::mouseMoveEvent(event);
}
