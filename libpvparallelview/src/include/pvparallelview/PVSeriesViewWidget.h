/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2018
 */

#ifndef __PVPARALLELVIEW_PVSERIESVIEWWIDGET_H__
#define __PVPARALLELVIEW_PVSERIESVIEWWIDGET_H__

#include <QWidget>
#include <QListWidget>

#include <inendi/PVView.h>

#include <pvkernel/widgets/PVHelpWidget.h>
#include <pvkernel/core/PVDisconnector.h>

namespace Inendi
{
class PVRangeSubSampler;
}

namespace PVParallelView
{

class PVSeriesView;
class PVSeriesViewZoomer;
class PVSeriesViewParamsWidget;

class PVSeriesViewWidget : public QWidget
{
	friend class PVSeriesViewParamsWidget;

  public:
	PVSeriesViewWidget(Inendi::PVView* view, PVCol axis, QWidget* parent = nullptr);

  protected:
	void keyPressEvent(QKeyEvent* event) override;
	void enterEvent(QEvent*) override;
	void leaveEvent(QEvent*) override;

  private:
	void set_abscissa(PVCol axis);
	void update_selected_series();
	bool is_in_region(QRect region, PVCol col) const;

  private:
	Inendi::PVView* _view;
	std::function<void(std::vector<QWidget*> const&)> _layout_replacer;
	std::unique_ptr<Inendi::PVRangeSubSampler> _sampler;
	PVSeriesView* _plot = nullptr;
	PVSeriesViewZoomer* _zoomer = nullptr;
	QListWidget* _series_list_widget = nullptr;

	bool _update_selected_series_resample = false;
	bool _synchro_selected_list = false;

	PVCore::PVDisconnector _plotting_change_connection;
	PVCore::PVDisconnector _selection_change_connection;

	PVSeriesViewParamsWidget* _params_widget;
	PVWidgets::PVHelpWidget _help_widget;
};

struct SerieListItemData {
	PVCol col;
	QColor color;
};
}

Q_DECLARE_METATYPE(PVParallelView::SerieListItemData)

#endif // __PVPARALLELVIEW_PVSERIESVIEWWIDGET_H__
