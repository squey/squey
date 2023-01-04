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

#ifndef PVGUIQT_PVTABLEVIEW_HPP
#define PVGUIQT_PVTABLEVIEW_HPP

#include <QTableView>

namespace PVGuiQt
{
/**
 * It is a QTableView with filtering on Tooltip event to display tooltip only
 * when the content can't be display in the corresponding cell.
 */
class PVTableView : public QTableView
{
	Q_OBJECT;

  public:
	explicit PVTableView(QWidget* parent) : QTableView(parent) {}

  Q_SIGNALS:
	/**
	 * Emit it on resize event.
	 */
	void resize();

  protected:
	/**
	 * Check for ToolTip event to disable tooltip when cell is big enough
	 * to show the full cell content
	 */
	bool viewportEvent(QEvent* event) override;

	/**
	 * Emite a resize signal on resize event.
	 */
	void resizeEvent(QResizeEvent* event) override;
};
} // namespace PVGuiQt

#endif
