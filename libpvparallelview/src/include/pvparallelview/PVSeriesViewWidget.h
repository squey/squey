/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2018
 */

#ifndef __PVPARALLELVIEW_PVSERIESVIEWWIDGET_H__
#define __PVPARALLELVIEW_PVSERIESVIEWWIDGET_H__

#include <QWidget>
#include <QTreeWidget>
#include <QStyledItemDelegate>

#include <pvparallelview/PVSeriesTreeWidget.h>
#include <inendi/PVView.h>
#include <inendi/PVRangeSubSampler.h>

#include <pvkernel/widgets/PVHelpWidget.h>
#include <pvkernel/core/PVDisconnector.h>

#include <pvcop/db/array.h>

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
	Q_OBJECT

	friend class PVSeriesViewParamsWidget;

  public:
	PVSeriesViewWidget(Inendi::PVView* view, PVCol axis, QWidget* parent = nullptr);
	~PVSeriesViewWidget() { delete _tree_model; }

  protected:
	void keyPressEvent(QKeyEvent* event) override;
	void enterEvent(QEvent*) override;
	void leaveEvent(QEvent*) override;

  private:
	void set_abscissa(PVCol axis);
	void set_split(PVCol axis);
	bool is_splitted() const { return _sampler->group_count() > 1; }
	void setup_series_tree(PVCol abscissa);
	void setup_selected_series_tree(PVCol abscissa);
	void update_selected_series();
	void synchro_list(QTreeWidget* list_src, QTreeWidget* list_dest);
	void semi_synchro_list(QTreeWidget* list_src, QTreeWidget* list_dest);
	bool is_in_region(const QRect region, PVCol col) const;

  private:
	Inendi::PVView* _view;
	std::function<void(std::vector<QWidget*> const&)> _layout_replacer;
	std::unique_ptr<Inendi::PVRangeSubSampler> _sampler;
	PVSeriesView* _plot = nullptr;
	PVSeriesViewZoomer* _zoomer = nullptr;
	PVSeriesTreeView* _series_tree_widget = nullptr;
	PVSeriesTreeView* _selected_series_tree = nullptr;
	QList<QTreeWidgetItem*> _selected_items;
	PVSeriesTreeModel* _tree_model = nullptr;
	QItemSelectionModel* _selection_model = nullptr;

	bool _update_selected_series_resample = false;
	bool _synchro_selected_list = false;

	PVCol _split_axis;

	PVCore::PVDisconnector _plotting_change_connection;
	PVCore::PVDisconnector _selection_change_connection;

	PVSeriesViewParamsWidget* _params_widget;
	PVWidgets::PVHelpWidget _help_widget;
};
}

#endif // __PVPARALLELVIEW_PVSERIESVIEWWIDGET_H__
