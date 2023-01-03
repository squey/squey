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

#ifndef _PVPARALLELVIEW_PVSERIESVIEWPARAMSWIDGET_H_
#define _PVPARALLELVIEW_PVSERIESVIEWPARAMSWIDGET_H_

#include <pvbase/types.h>
#include <pvparallelview/PVSeriesView.h>

#include <QToolBar>

class QToolButton;
class QSignalMapper;

namespace PVWidgets
{
class PVAxisComboBox;
}

namespace PVParallelView
{

class PVSeriesViewWidget;

class PVSeriesViewParamsWidget : public QToolBar
{
	Q_OBJECT

  public:
	PVSeriesViewParamsWidget(PVCol abscissa, PVSeriesViewWidget* parent);

  private:
	void add_abscissa_selector(PVCol axis);
	void add_split_selector();
	QToolButton* add_rendering_mode_selector();
	QToolButton* add_sampling_mode_selector();
	void add_selection_activator(bool enable);
	void add_hunting_activator(bool enable);

	void set_rendering_mode(QAction* action);
	void set_rendering_mode(PVSeriesView::DrawMode mode) const;
	void set_rendering_mode();
	void set_sampling_mode(QAction* action);
	void set_sampling_mode(Inendi::PVRangeSubSampler::SAMPLING_MODE mode) const;
	void set_sampling_mode();

	void update_mode_selector(QToolButton* button, int mode_index);
	void change_abscissa(PVCol abscissa);

  private:
	QToolButton* _rendering_mode_button = nullptr;
	QToolButton* _sampling_mode_button = nullptr;
	PVSeriesViewWidget* _series_view_widget = nullptr;
	PVWidgets::PVAxisComboBox* _abscissa_selector = nullptr;
	std::vector<std::function<void()>> _bind_connections;
	PVSeriesView::DrawMode _rendering_mode;
	Inendi::PVRangeSubSampler::SAMPLING_MODE _sampling_mode;
};

} // namespace PVParallelView

#endif // _PVPARALLELVIEW_PVSERIESVIEWPARAMSWIDGET_H_
