/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef _PVPARALLELVIEW_PVSERIESVIEWPARAMSWIDGET_H_
#define _PVPARALLELVIEW_PVSERIESVIEWPARAMSWIDGET_H_

#include <pvbase/types.h>
#include <pvparallelview/PVSeriesView.h>

#include <QToolBar>

class QToolButton;
class QSignalMapper;

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
	PVSeriesViewWidget* _series_view_widget;
	std::vector<std::function<void()>> _bind_connections;
	PVSeriesView::DrawMode _rendering_mode;
	Inendi::PVRangeSubSampler::SAMPLING_MODE _sampling_mode;
};

} // namespace PVParallelView

#endif // _PVPARALLELVIEW_PVSERIESVIEWPARAMSWIDGET_H_