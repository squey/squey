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

#ifndef PVPARALLELVIEW_PVZOOMEDPARALLELVIEWPARAMSWIDGET_H
#define PVPARALLELVIEW_PVZOOMEDPARALLELVIEWPARAMSWIDGET_H

#include <QToolBar>

#include <pvbase/types.h>
#include <squey/widgets/PVAxisComboBox.h>

#include <QStringList>
class QMenu;
class QToolButton;

namespace PVParallelView
{
class PVZoomedParallelView;

class PVZoomedParallelViewParamsWidget : public QToolBar
{
	Q_OBJECT

  public:
	explicit PVZoomedParallelViewParamsWidget(Squey::PVAxesCombination const& axes_comb,
	                                          QWidget* parent);

  public:
	void build_axis_menu(PVCombCol active_axis);

  Q_SIGNALS:
	void change_to_col(PVCombCol new_axis);

  private:
	PVWidgets::PVAxisComboBox* _menu;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_ZOOMEDPARALLELVIEWPARAMSWIDGET_H
