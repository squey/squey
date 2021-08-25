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

#ifndef PVPARALLELVIEW_PVFULLPARALLELVIEWPARAMSWIDGET_H
#define PVPARALLELVIEW_PVFULLPARALLELVIEWPARAMSWIDGET_H

#include <QToolBar>

class QToolBar;
class QCheckBox;
class QSignalMapper;
class QToolButton;

namespace PVParallelView
{

class PVFullParallelView;

class PVFullParallelViewParamsWidget : public QToolBar
{
	Q_OBJECT

  public:
	explicit PVFullParallelViewParamsWidget(PVFullParallelView* parent);

  public:
	void update_widgets();

  private Q_SLOTS:
	void set_selection_mode(int mode);

  private:
	PVFullParallelView* parent_fpv() const;

  private:
	QAction* _autofit;
	QAction* _use_log_color;
	QAction* _show_labels;
	QSignalMapper* _signal_mapper;
	QToolButton* _sel_mode_button;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVFULLPARALLELVIEWPARAMSWIDGET_H
