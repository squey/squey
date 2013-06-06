
#include <pvkernel/widgets/PVPopupWidget.h>

#include <QApplication>
#include <QDesktopWidget>
#include <QMouseEvent>

#define AlignHoriMask (PVWidgets::PVPopupWidget::AlignLeft | PVWidgets::PVPopupWidget::AlignRight | PVWidgets::PVPopupWidget::AlignHCenter)
#define AlignVertMask (PVWidgets::PVPopupWidget::AlignTop | PVWidgets::PVPopupWidget::AlignBottom | PVWidgets::PVPopupWidget::AlignVCenter)

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
	QDialog(parent)
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
 * PVWidgets::PVPopupWidget::popup
 *****************************************************************************/

void PVWidgets::PVPopupWidget::popup(QWidget* widget,
                                     int align, int expand,
                                     int border, bool fit_in_screen)
{
	if (isVisible()) {
		return;
	}

	// make sure the popup's geometry is correct
	adjustSize();

	QRect parent_geom = widget->geometry();

	/* make sure to have coordinates which are relative
	 * to the screen, and not relative to the parent
	 * widget.
	 */
	parent_geom.moveTo(widget->mapToGlobal(parent_geom.topLeft()));

	/* about borders
	 */
	parent_geom = QRect(parent_geom.x() + border,
	                    parent_geom.y() + border,
	                    parent_geom.width() - 2 * border,
	                    parent_geom.height() - 2 * border);

	QRect current_geom = geometry();
	QPoint center_pos = parent_geom.center();
	QRect new_geom;

 	if (expand & ExpandX) {
		new_geom.setWidth(parent_geom.width());
	} else {
		new_geom.setWidth(current_geom.width());
	}

	if (expand & ExpandY) {
		new_geom.setHeight(parent_geom.height());
	} else {
		new_geom.setHeight(current_geom.height());
	}

	switch(align & AlignHoriMask) {
	case AlignRight:
		new_geom.moveLeft(parent_geom.right() - new_geom.width());
		break;
	case AlignHCenter:
		new_geom.moveLeft(center_pos.x() - new_geom.width() / 2);
		break;
	case AlignLeft:
	default:
		new_geom.moveLeft(parent_geom.left());
		break;
	}
	switch(align & AlignVertMask) {
	case AlignBottom:
		new_geom.moveTop(parent_geom.bottom() - new_geom.height());
		break;
	case AlignVCenter:
		new_geom.moveTop(center_pos.y() - new_geom.height() / 2);
		break;
	case AlignTop:
	default:
		new_geom.moveTop(parent_geom.top());
		break;
	}

	if (fit_in_screen) {
		setGeometry(fitToScreen(new_geom));
	} else {
		setGeometry(new_geom);
	}
	raise();
	show();
}

/*****************************************************************************
 * PVWidgets::PVPopupWidget::setVisible
 *****************************************************************************/

void PVWidgets::PVPopupWidget::setVisible(bool visible)
{
	QDialog::setVisible(visible);

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
	/* RH: the initial problem is that the PVPopupWidget in mode Qt::Popup
	 * badly propagate wheel event to its parent when it is made hidden:
	 * - QTBUG-27478 reports that the Qt::WA_UnderMouse QWidget's
	 *   attribute is wrongly updated in case of popup widget closure;
	 *   that problem leaves Qt in an internally bad state solved by an
	 *   update of the Qt::WA_UnderMouse attribute (like clicking in an
	 *   other widget)
	 * - the last mouse position stored by the PV-I view is erroneous
	 *   because the mouse pointer is "teleported" accross the popup
	 *
	 * using the fix for QTBUG-27478 does not solve anything as the mouse
	 * pointer is still teleporting accross the popup. The lone solution
	 * is to make sure that:
	 * - the popup widget forwards, at least, all mouse movements to its
	 *   parent, so that it can update its internal state
	 * - the popup mode give the Qt::WA_UnderMouse to its parent when the
	 *   mouse pointer is outside of its geometry
	 *
	 * The good point is that Qt does not have to be patch as the
	 * Qt::WA_UnderMouse attribute is necessarly updated.
	 */
	if (geometry().contains(mapFromGlobal(QCursor::pos()))) {
		parentWidget()->setAttribute(Qt::WA_UnderMouse, false);
	} else {
		parentWidget()->setAttribute(Qt::WA_UnderMouse, true);
	}

	QMouseEvent pevent(QEvent::MouseMove,
	                   parentWidget()->mapFromGlobal(event->globalPos()),
	                   event->globalPos(),
	                   event->button(),
	                   event->buttons(),
	                   event->modifiers());

	QApplication::sendEvent(parentWidget(), &pevent);
	event->ignore();
	QDialog::mouseMoveEvent(event);
}
