/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef __PVGUIQT_PVABSTRACTLISTSTATSDLG_H__
#define __PVGUIQT_PVABSTRACTLISTSTATSDLG_H__

#include <pvkernel/rush/PVNraw.h>

#include <inendi/PVView_types.h>

#include <pvguiqt/PVListDisplayDlg.h>
#include <pvguiqt/PVStatsModel.h> // TODO : remove it

#include <QStyledItemDelegate>

#include <QResizeEvent>

class QComboBox;

namespace PVGuiQt {

class PVStatsModel;

namespace __impl {
class PVListStringsDelegate;
class PVAbstractListStatsRangePicker;
}

class PVStatsSortProxyModel;

class PVAbstractListStatsDlg: public PVListDisplayDlg
{
	// TODO: Better members visibility
	Q_OBJECT

	friend class __impl::PVListStringsDelegate;

public:
	PVAbstractListStatsDlg(
		Inendi::PVView_sp& view,
		PVCol c,
		PVStatsModel* model,
		double absolute_max_count,
		double relative_min_count,
		double relative_max_count,
		QWidget* parent = nullptr
	);

	void init(Inendi::PVView_sp& view);

public:
	inline double absolute_max_count() const { return ((PVStatsModel const*)model())->absolute_max_count(); }
	inline double relative_max_count() const { return ((PVStatsModel const*)model())->relative_max_count(); }
	inline double max_count() const { return ((PVStatsModel const*)model())->max_count(); }

	inline bool use_logarithmic_scale() { return ((PVStatsModel const*)model())->use_log_scale();  }

protected:
	void showEvent(QShowEvent * event) override;
	void sort_by_column(int col);
	bool process_context_menu(QAction* act) override;
	void ask_for_copying_count() override;

	/**
	 * create a new layer using the selected values.
	 */
	void create_layer_with_selected_values();

	/**
	 * create a set of new layers, one layer for each selected value.
	 */
	void create_layers_for_selected_values();

protected slots:
	void view_resized();
	void section_resized(int logicalIndex, int oldSize, int newSize);
	void scale_changed(QAction* act);
	void max_changed(QAction* act);
	void section_clicked(int col);
	void sort();

protected slots:
	void select_set_mode_count(bool checked);
	void select_set_mode_frequency(bool checked);
	void select_refresh(bool checked);

	/**
	 * Show the context menu on header view for the second column.
	 */
	void show_hhead_ctxt_menu(const QPoint& pos);

protected:
	Inendi::PVView* lib_view() { return _obs.get_object(); }
	void multiple_search(QAction* act, const QStringList &sl, bool hide_dialog = true);
	void resize_section();

protected:
	PVCol _col;
	PVHive::PVObserverSignal<Inendi::PVView> _obs;
	PVHive::PVActor<Inendi::PVView> _actor;
	bool _store_last_section_width = true;
	int _last_section_width = 250;

	QAction* _act_toggle_linear;
	QAction* _act_toggle_log;
	QAction* _act_toggle_absolute;
	QAction* _act_toggle_relative;

	QAction* _act_show_percentage;
	QAction* _act_show_count;
	QAction* _act_show_scientific_notation;

	__impl::PVAbstractListStatsRangePicker* _select_picker;
	bool                                    _select_is_count;

	QMenu* _copy_values_menu;
	QAction* _copy_values_without_count_act;
	QAction* _copy_values_with_count_act;

	QAction* _create_layer_with_values_act;
	QAction* _create_layers_for_values_act;

	QMenu* _hhead_ctxt_menu; //!< Context menu for right click on the vertical headers

private:
	/**
	 * RH: a litle hack to replace a bigger one :-]
	 * We use the multiple search action named "Search for this value" to
	 * create layer(s) from selected values.
	 * But as those actions come from a plugins, we have to retrieve it
	 * while filing the context menu.
	 */
	QAction* _msearch_action_for_layer_creation;
};

namespace __impl {

class PVListStringsDelegate: public QStyledItemDelegate
{
	Q_OBJECT

public:
	PVListStringsDelegate(PVAbstractListStatsDlg* parent) : QStyledItemDelegate(parent) {}

protected:
	void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override;

	PVGuiQt::PVAbstractListStatsDlg* d() const;
};

}

}


#endif // __PVGUIQT_PVABSTRACTLISTSTATSDLG_H__
