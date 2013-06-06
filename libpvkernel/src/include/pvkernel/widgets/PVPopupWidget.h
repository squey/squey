
#ifndef PVWIDGETS_PVPOPUPWIDGET_H
#define PVWIDGETS_PVPOPUPWIDGET_H

#include <QDialog>

namespace PVWidgets
{

/**
 * a generic popup widget to display any widget set over an QWidget
 *
 * @note RH: when in Qt::Popup mode, this widget had a problem when
 * it was made hidden after a wheel event: the parent widget's attribute
 * Qt::WA_UnderMouse was not correctly updated by Qt. QTBUG-27478 reports
 * that problem (solved in Qt5 but not backported to Qt4). Patching Qt does
 * not seem to solve entirely our problem (we maybe expect a behaviour that
 * is not standard). So that overriding {enter,leave}Event to force the
 * parent widget's Qt::WA_UnderMouse attribute to a correct value do what
 * we want.
 *
 * @note RH: when the popup is visible, a wheel event outside the popup
 * do a weird zoom. It can be solved by forwarding mouse move events from
 * the popup widget to its parent.
 */

class PVPopupWidget : public QDialog
{
public:
	/**
	 * a orizable enumeration to tell how a popup will be placed on screen
	 */
	typedef enum {
		AlignNone       =  0,
		AlignLeft       =  1,
		AlignRight      =  2,
		AlignHCenter    =  4,
		AlignTop        =  8,
		AlignBottom     = 16,
		AlignVCenter    = 32,
		AlignUnderMouse = 64,
		AlignCenter     = AlignHCenter + AlignVCenter
	} AlignEnum;

	/**
	 * a orizable enumeration to tell how a popup is expanded
	 */
	typedef enum {
		ExpandNone = 0,
		ExpandX    = 1,
		ExpandY    = 2,
		ExpandAll  = ExpandX + ExpandY
	} ExpandEnum;

public:
	/**
	 * create a new popup widget
	 *
	 * @note contrary to Qt's widgets, a parent is required
	 *
	 * @param parent the parent QWidget
	 */
	PVPopupWidget(QWidget* parent);

public:
	/**
	 * make the popup visible at screen coord
	 *
	 * @param p the position on screen
	 * @param centered set to true to have the popup centered on p; false if the upper left
	 * corner must be set to p
	 */
	void popup(const QPoint& p, bool centered = false);

	void popup(QWidget* widget, int align = AlignNone, int expand = ExpandNone,
	           int border = 0, bool fit_in_screen = false);

	/**
	 * reimplement QDialog::setVisible(bool)
	 *
	 * to move focus from parent to poup and from popup to parent
	 */
	void setVisible(bool visible) override;

protected:
	/**
	 * reimplement QDialog::mouseMoveEvent(QMouseEvent)
	 *
	 * to force its parent's attribute Qt::WA_UnderMouse to be false
	 */
	void enterEvent(QEvent* event) override;

	/**
	 * reimplement QDialog::mouseMoveEvent(QMouseEvent)
	 *
	 * to force its parent's attribute Qt::WA_UnderMouse to be true
	 */
	void leaveEvent(QEvent* event) override;

	/**
	 * reimplement QDialog::mouseMoveEvent(QMouseEvent)
	 *
	 * to forward mouse move event to its parent.
	 */
	void mouseMoveEvent(QMouseEvent* event) override;
};

}

#endif // PVWIDGETS_PVPOPUPWIDGET_H
