
#ifndef PVWIDGETS_PVPOPUPWIDGET_H
#define PVWIDGETS_PVPOPUPWIDGET_H

#include <QWidget>

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
 * parent widget's Qt::WA_UnderMouse attribute to a correct value that do
 * what we want.
 *
 * @note RH: when the popup is visible, a wheel event outside the popup
 * do a weird zoom. It can be solved by forwarding mouse move events from
 * the popup widget to its parent.
 */

class PVPopupWidget : public QWidget
{
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

	/**
	 * reimplement QDialog::setVisible(bool)
	 *
	 * to move focus from parent to popup and from popup to parent
	 */
	void setVisible(bool visible) override;

public:
	/**
	 * test if key is one of those to close the widget
	 *
	 * This method has to be overridden to accept more key
	 *
	 * @param key the key to test
	 *
	 * @return true is key can be used to close the widget; false otherwise.
	 */
	virtual bool is_close_key(int key);

protected:
	/**
	 * reimplement QDialog::mouseMoveEvent(QMouseEvent)
	 *
	 * to forward mouse move event to its parent.
	 */
	void mouseMoveEvent(QMouseEvent* event) override;
};

}

#endif // PVWIDGETS_PVPOPUPWIDGET_H
