/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

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
	explicit PVPopupWidget(QWidget* parent);

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
} // namespace PVWidgets

#endif // PVWIDGETS_PVPOPUPWIDGET_H
