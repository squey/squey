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

#ifndef __PVPARALLELVIEW_PVSERIESVIEWWIDGET_H__
#define __PVPARALLELVIEW_PVSERIESVIEWWIDGET_H__

#include <QWidget>
#include <QTreeWidget>
#include <QStyledItemDelegate>

#include <pvparallelview/PVSeriesTreeWidget.h>
#include <squey/PVView.h>
#include <squey/PVRangeSubSampler.h>

#include <pvkernel/widgets/PVHelpWidget.h>
#include <pvkernel/widgets/PVMouseButtonsLegend.h>
#include <pvkernel/core/PVDisconnector.h>

#include <pvcop/db/array.h>

namespace Squey
{
class PVRangeSubSampler;
}

namespace PVWidgets
{
class PVRangeEdit;
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
	PVSeriesViewWidget(Squey::PVView* view, PVCol axis, QWidget* parent = nullptr);
	~PVSeriesViewWidget() { delete _tree_model; }

  protected:
	void keyPressEvent(QKeyEvent* event) override;
	void keyReleaseEvent(QKeyEvent* event) override;
	void enterEvent(QEnterEvent*) override;
	void leaveEvent(QEvent*) override;

  private:
	void setup_layout();
	void update_layout();
	void update_window_title(PVCol axis);
	void set_abscissa(PVCol axis);
	void set_split(PVCol axis);
	bool is_splitted() const { return _sampler->group_count() > 1; }
	void setup_series_tree(PVCol abscissa);
	void setup_selected_series_tree(PVCol abscissa);
	void update_selected_series();
	bool is_in_region(const QRect region, PVCol col) const;
	void minmax_changed(const pvcop::db::array& minmax);
	void select_all_series(bool use_axes_combination = true);

	/**
	 * Shows, next to each drawn series, its values in the sample under the cursor, or in the
	 * closest one within the hole of the crosshair: the value itself, or their range when the
	 * sample holds several.
	 */
	void update_cursor_values();

	/**
	 * Lists only the selected series, those the plot draws, while @p selected_only; all of them
	 * otherwise.
	 */
	void show_selected_series_only(bool selected_only);

	/**
	 * Hides the rows of the series left out of the selection while only the selected ones are
	 * listed, a column once none of its groups is selected; shows every row otherwise.
	 */
	void update_hidden_series();

  Q_SIGNALS:
	void set_status_bar_mouse_legend(PVWidgets::PVMouseButtonsLegend legend);
	void clear_status_bar_mouse_legend();

  private:
	Squey::PVView* _view;
	std::unique_ptr<Squey::PVRangeSubSampler> _sampler;
	PVSeriesView* _plot = nullptr;
	PVWidgets::PVRangeEdit* _range_edit = nullptr;
	PVSeriesViewZoomer* _zoomer = nullptr;
	PVSeriesTreeView* _series_tree_widget = nullptr;
	PVSeriesTreeView* _selected_series_tree = nullptr;
	QList<QTreeWidgetItem*> _selected_items;
	PVSeriesTreeModel* _tree_model = nullptr;
	QItemSelectionModel* _selection_model = nullptr;

	bool _update_selected_series_resample = false;
	bool _synchro_selected_list = false;
	bool _selected_series_only = false;

	PVCol _abscissa_axis;
	PVCol _split_axis;

	int _cursor_x = -1; //!< The column of pixels of _zoomer under the cursor, -1 out of it.

	PVCore::PVDisconnector _scaling_change_connection;
	PVCore::PVDisconnector _selection_change_connection;
	PVCore::PVDisconnector _view_deleted_connection;

	PVSeriesViewParamsWidget* _params_widget = nullptr;
	PVWidgets::PVHelpWidget _help_widget;

	std::vector<std::function<void()>> _updaters;

	PVWidgets::PVMouseButtonsLegend _mouse_buttons_current_legend;
	PVWidgets::PVMouseButtonsLegend _mouse_buttons_default_legend;
};
} // namespace PVParallelView

#endif // __PVPARALLELVIEW_PVSERIESVIEWWIDGET_H__
