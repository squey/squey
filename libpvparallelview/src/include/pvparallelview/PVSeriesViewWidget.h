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
	PVSeriesViewWidget(Inendi::PVView* view, PVCombCol axis_comb, QWidget* parent = nullptr);

  protected:
	void keyPressEvent(QKeyEvent* event) override;
	void enterEvent(QEvent*) override;
	void leaveEvent(QEvent*) override;

  private:
	void update_selected_series();
	bool is_in_region(QRect region, PVCol col) const;

  private:
	std::unique_ptr<Inendi::PVRangeSubSampler> _sampler;
	PVSeriesView* _plot;
	PVSeriesViewZoomer* _zoomer;
	QListWidget* _series_list_widget;

	bool _update_selected_series_resample = false;
	bool _synchro_selected_list = false;

	struct Disconnector : public sigc::connection {
		using sigc::connection::connection;
		using sigc::connection::operator=;
		Disconnector(Disconnector&&) = delete;
		~Disconnector() { disconnect(); }
	};

	Disconnector _plotting_change_connection;
	Disconnector _selection_change_connection;

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