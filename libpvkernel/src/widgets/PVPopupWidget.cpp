
#include <pvkernel/widgets/PVPopupWidget.h>

#include <QApplication>
#include <QDesktopWidget>

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
