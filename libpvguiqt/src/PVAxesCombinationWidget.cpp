/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVAxesCombination.h>
#include <inendi/PVView.h>
#include <inendi/PVPlotted.h>
#include <pvguiqt/PVAxesCombinationWidget.h>

#include <QDialogButtonBox>

PVGuiQt::PVAxesCombinationWidget::PVAxesCombinationWidget(
    Inendi::PVAxesCombination& axes_combination, Inendi::PVView* view, QWidget* parent)
    : QWidget(parent), _axes_combination(axes_combination), _view(view)
{
	setupUi(this);

	update_all();

	connect(_btn_axis_add, SIGNAL(clicked()), this, SLOT(axis_add_Slot()));
	connect(_btn_axis_up, SIGNAL(clicked()), this, SLOT(axis_up_Slot()));
	connect(_btn_axis_down, SIGNAL(clicked()), this, SLOT(axis_down_Slot()));
	connect(_btn_axis_remove, SIGNAL(clicked()), this, SLOT(axis_remove_Slot()));
	connect(_btn_sort, SIGNAL(clicked()), this, SLOT(sort_Slot()));
	connect(_btn_reset, SIGNAL(clicked()), this, SLOT(reset_comb_Slot()));

	_btn_sel_singleton->setEnabled(view != nullptr);
	if (view != nullptr) {
		connect(_btn_sel_singleton, SIGNAL(clicked()), this, SLOT(sel_singleton_Slot()));
	}
}

void PVGuiQt::PVAxesCombinationWidget::axis_add_Slot()
{
	if (!is_original_axis_selected()) {
		return;
	}

	PVCol axis_id = get_original_axis_selected();
	QString axis_name = get_original_axis_selected_name();

	_list_used->addItem(axis_name);
	_axes_combination.axis_append(axis_id);

	_list_used->setCurrentRow(_list_used->count() - 1);
}

void PVGuiQt::PVAxesCombinationWidget::axis_up_Slot()
{
	if (!is_used_axis_selected()) {
		return;
	}

	QVector<PVCol> axes_id(get_used_axes_selected());
	for (PVCol c : axes_id) {
		if (c == 0) {
			return;
		}
	}

	_axes_combination.move_axes_left_one_position(axes_id.begin(), axes_id.end());
	update_used_axes();
	QItemSelection new_sel;
	for (PVCol c : axes_id) {
		QModelIndex midx = _list_used->model()->index(c - 1, 0);
		new_sel.select(midx, midx);
	}
	_list_used->selectionModel()->select(new_sel, QItemSelectionModel::ClearAndSelect);
}

void PVGuiQt::PVAxesCombinationWidget::axis_down_Slot()
{
	if (!is_used_axis_selected()) {
		return;
	}

	QVector<PVCol> axes_id(get_used_axes_selected());
	for (PVCol c : axes_id) {
		if (c == _list_used->count() - 1) {
			return;
		}
	}

	_axes_combination.move_axes_right_one_position(axes_id.begin(), axes_id.end());
	update_used_axes();
	QItemSelection new_sel;
	for (PVCol c : axes_id) {
		QModelIndex midx = _list_used->model()->index(c + 1, 0);
		new_sel.select(midx, midx);
	}
	_list_used->selectionModel()->select(new_sel, QItemSelectionModel::ClearAndSelect);
}

void PVGuiQt::PVAxesCombinationWidget::axis_remove_Slot()
{
	if (!is_used_axis_selected()) {
		return;
	}

	// We need a minimum of 2 axes !
	if (_list_used->count() <= 2) {
		return;
	}

	QVector<PVCol> axes_id = get_used_axes_selected();
	_axes_combination.remove_axes(axes_id.begin(), axes_id.end());
	update_used_axes();
	_list_used->setCurrentRow(std::min((int)axes_id.at(0), _list_used->count() - 1));
}

void PVGuiQt::PVAxesCombinationWidget::reset_comb_Slot()
{
	_axes_combination.reset_to_default();

	update_used_axes();
}

PVCol PVGuiQt::PVAxesCombinationWidget::get_original_axis_selected()
{
	return _list_org->currentRow();
}

QString PVGuiQt::PVAxesCombinationWidget::get_original_axis_selected_name()
{
	return _list_org->currentItem()->text();
}

QVector<PVCol> PVGuiQt::PVAxesCombinationWidget::get_list_selection(QListWidget* widget)
{
	QVector<PVCol> ret;
	QModelIndexList list = widget->selectionModel()->selectedIndexes();
	ret.reserve(list.size());
	for (const QModelIndex& idx : list) {
		ret.push_back(idx.row());
	}
	return ret;
}

QVector<PVCol> PVGuiQt::PVAxesCombinationWidget::get_used_axes_selected()
{
	return get_list_selection(_list_used);
}

void PVGuiQt::PVAxesCombinationWidget::update_orig_axes()
{
	_list_org->clear();
	_list_org->addItems(_axes_combination.get_nraw_names());
}

void PVGuiQt::PVAxesCombinationWidget::update_used_axes()
{
	_list_used->clear();
	_list_used->addItems(_axes_combination.get_combined_names());
}

void PVGuiQt::PVAxesCombinationWidget::update_all()
{
	update_orig_axes();
	update_used_axes();
}

void PVGuiQt::PVAxesCombinationWidget::reset_used_axes()
{
	_axes_combination.set_combination(_view->get_axes_combination().get_combination());
	update_used_axes();
}

void PVGuiQt::PVAxesCombinationWidget::sort_Slot()
{
	_axes_combination.sort_by_name();
	update_used_axes();
}

bool PVGuiQt::PVAxesCombinationWidget::is_used_axis_selected()
{
	return _list_used->selectedItems().size() > 0;
}

bool PVGuiQt::PVAxesCombinationWidget::is_original_axis_selected()
{
	return _list_org->selectedItems().size() > 0;
}

void PVGuiQt::PVAxesCombinationWidget::set_selection_from_cols(QList<PVCol> const& cols)
{
	QItemSelection new_sel;
	for (size_t comb_col = 0; comb_col < _axes_combination.get_axes_count(); comb_col++) {
		if (cols.contains(_axes_combination.get_combination()[comb_col])) {
			QModelIndex midx = _list_used->model()->index(comb_col, 0);
			new_sel.select(midx, midx);
		}
	}
	_list_used->selectionModel()->select(new_sel, QItemSelectionModel::ClearAndSelect);
}

void PVGuiQt::PVAxesCombinationWidget::sel_singleton_Slot()
{
	assert(_view);
	QList<PVCol> cols_rem = _view->get_parent<Inendi::PVPlotted>().get_singleton_columns_indexes();
	set_selection_from_cols(cols_rem);
}
