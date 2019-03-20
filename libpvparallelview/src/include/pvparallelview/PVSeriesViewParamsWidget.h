/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef _PVPARALLELVIEW_PVSERIESVIEWPARAMSWIDGET_H_
#define _PVPARALLELVIEW_PVSERIESVIEWPARAMSWIDGET_H_

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
	PVSeriesViewParamsWidget(PVSeriesViewWidget* parent);

  private:
	QToolButton* add_rendering_mode_selector();
	QToolButton* add_sampling_mode_selector();
	void add_selection_activator();
	void add_hunting_activator();

	void set_rendering_mode(QAction* action);
	void set_rendering_mode();
	void set_sampling_mode(QAction* action);
	void set_sampling_mode();

	void update_mode_selector(QToolButton* button, int mode_index);

  private:
	QToolButton* _rendering_mode_button;
	QToolButton* _sampling_mode_button;
	PVSeriesViewWidget* _series_view_widget;
};

} // namespace PVParallelView

#endif // _PVPARALLELVIEW_PVSERIESVIEWPARAMSWIDGET_H_