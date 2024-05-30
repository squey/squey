//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <squey/PVAxesCombination.h>
#include <squey/PVView.h>
#include <squey/PVScaled.h>
#include <pvguiqt/PVAxesCombinationWidget.h>
#include <pvkernel/widgets/PVModdedIcon.h>

#include <QDialogButtonBox>
#include <QDebug>
#include <QCloseEvent>

#include <unordered_set>

PVGuiQt::PVAxesCombinationWidget::PVAxesCombinationWidget(
    Squey::PVAxesCombination& axes_combination, Squey::PVView* view, QWidget* parent)
    : QWidget(parent), _axes_combination(axes_combination), _view(view)
{
	setupUi(this);

	update_all();

	enable_drop(true);

	connect(_btn_reset, &QAbstractButton::clicked, this, &PVAxesCombinationWidget::reset_comb_Slot);
	connect(_btn_axis_add_before, &QAbstractButton::clicked, this,
	        &PVAxesCombinationWidget::axis_add_before_Slot);
	connect(_btn_axis_add_after, &QAbstractButton::clicked, this,
	        &PVAxesCombinationWidget::axis_add_after_Slot);
	connect(_btn_select_all, &QAbstractButton::clicked, this,
	        &PVAxesCombinationWidget::select_all_Slot);
	connect(_btn_invert_selection, &QAbstractButton::clicked, this,
	        &PVAxesCombinationWidget::invert_selection_Slot);
	connect(_btn_axis_top, &QAbstractButton::clicked, this,
	        &PVAxesCombinationWidget::move_top_Slot);
	connect(_btn_axis_up, &QAbstractButton::clicked, this, &PVAxesCombinationWidget::axis_up_Slot);
	connect(_btn_gather, &QAbstractButton::clicked, this,
	        &PVAxesCombinationWidget::gather_selected_Slot);
	connect(_btn_axis_down, &QAbstractButton::clicked, this,
	        &PVAxesCombinationWidget::axis_down_Slot);
	connect(_btn_axis_bottom, &QAbstractButton::clicked, this,
	        &PVAxesCombinationWidget::move_bottom_Slot);
	connect(_btn_axis_remove, &QAbstractButton::clicked, this,
	        &PVAxesCombinationWidget::axis_remove_Slot);
	connect(_btn_remove_duplicates, &QAbstractButton::clicked, this,
	        &PVAxesCombinationWidget::remove_duplicates_Slot);
	connect(_btn_sort, &QAbstractButton::clicked, this, &PVAxesCombinationWidget::sort_Slot);

	_btn_sel_singleton->setEnabled(view != nullptr);
	if (view != nullptr) {
		connect(_btn_sel_singleton, &QAbstractButton::clicked, this,
		        &PVAxesCombinationWidget::sel_singleton_Slot);
	}

	_btn_select_all->setIcon(PVModdedIcon("square-check"));
	_btn_invert_selection->setIcon(PVModdedIcon("square-check-inverted"));
	_btn_sel_singleton->setIcon(PVModdedIcon("square-1"));
	_btn_axis_top->setIcon(PVModdedIcon("arrow-up-to-line"));
	_btn_axis_up->setIcon(PVModdedIcon("arrow-up-long"));
	_btn_gather->setIcon(PVModdedIcon("arrows-minimize"));
	_btn_axis_down->setIcon(PVModdedIcon("arrow-down-long"));
	_btn_axis_bottom->setIcon(PVModdedIcon("arrow-down-to-line"));
	_btn_axis_remove->setIcon(PVModdedIcon("trash-xmark"));
	_btn_remove_duplicates->setIcon(PVModdedIcon("copy-x"));
	_btn_sort->setIcon(PVModdedIcon("arrow-down-short-wide"));
	_btn_reset->setIcon(PVModdedIcon("arrow-rotate-left"));
	_btn_axis_add_before->setIcon(PVModdedIcon("insert-before"));
	_btn_axis_add_after->setIcon(PVModdedIcon("insert-after"));
}

void PVGuiQt::PVAxesCombinationWidget::reset_comb_Slot()
{
	_axes_combination.reset_to_default();

	update_used_axes();
}

void PVGuiQt::PVAxesCombinationWidget::axis_add_before_Slot()
{
	DisableDnD ddd(this);
	auto selected_org = ordered_selected(_list_org);
	if (selected_org.empty()) {
		return;
	}
	int insert_row = 0;
	auto selected_used = ordered_selected(_list_used);
	if (not selected_used.empty()) {
		insert_row = _list_used->row(selected_used.front());
	}
	_list_used->clearSelection();
	for (auto item_org : selected_org) {
		auto item_used = new QListWidgetItem(*item_org);
		_list_used->insertItem(insert_row++, item_used);
		item_used->setSelected(true);
	}
	_list_used->scrollToItem(ordered_selected(_list_used).front());
	update_combination();
}

void PVGuiQt::PVAxesCombinationWidget::axis_add_after_Slot()
{
	DisableDnD ddd(this);
	auto selected_org = ordered_selected(_list_org);
	if (selected_org.empty()) {
		return;
	}
	int insert_row = _list_used->count();
	auto selected_used = ordered_selected(_list_used);
	if (not selected_used.empty()) {
		insert_row = _list_used->row(selected_used.back()) + 1;
	}
	_list_used->clearSelection();
	for (auto item_org : selected_org) {
		auto item_used = new QListWidgetItem(*item_org);
		_list_used->insertItem(insert_row++, item_used);
		item_used->setSelected(true);
	}
	_list_used->scrollToItem(ordered_selected(_list_used).back());
	update_combination();
}

void PVGuiQt::PVAxesCombinationWidget::select_all_Slot()
{
	_list_used->selectAll();
}

void PVGuiQt::PVAxesCombinationWidget::invert_selection_Slot()
{
	for (int i = 0; i < _list_used->count(); ++i) {
		auto item = _list_used->item(i);
		item->setSelected(not item->isSelected());
	}
}

void PVGuiQt::PVAxesCombinationWidget::axis_up_Slot()
{
	DisableDnD ddd(this);
	auto selected_used = ordered_selected(_list_used);
	if (selected_used.empty()) {
		return;
	}
	if (not std::all_of(selected_used.begin(), selected_used.end(),
	                    [this](auto item) { return _list_used->row(item) > 0; })) {
		return;
	}
	for (auto item_used : selected_used) {
		auto item_row = _list_used->row(item_used);
		_list_used->insertItem(item_row - 1, _list_used->takeItem(item_row));
	}
	for (auto item : selected_used) {
		item->setSelected(true);
	}
	update_combination();
}

void PVGuiQt::PVAxesCombinationWidget::axis_down_Slot()
{
	DisableDnD ddd(this);
	const auto& list_used = ordered_selected(_list_used);
	auto selected_used = std::list<QListWidgetItem*>(list_used.begin(), list_used.end());
	if (selected_used.empty()) {
		return;
	}
	if (not std::all_of(selected_used.begin(), selected_used.end(), [this](auto item) {
		    return _list_used->row(item) < _list_used->count() - 1;
		})) {
		return;
	}
	std::for_each(selected_used.rbegin(), selected_used.rend(), [this](auto item_used) {
		auto item_row = _list_used->row(item_used);
		_list_used->insertItem(item_row + 1, _list_used->takeItem(item_row));
	});
	for (auto item : selected_used) {
		item->setSelected(true);
	}
	update_combination();
}

void PVGuiQt::PVAxesCombinationWidget::move_top_Slot()
{
	DisableDnD ddd(this);
	auto selected_used = ordered_selected(_list_used);
	if (selected_used.empty()) {
		return;
	}
	int insert_row = 0;
	for (auto item_used : selected_used) {
		auto item_row = _list_used->row(item_used);
		_list_used->insertItem(insert_row++, _list_used->takeItem(item_row));
	}
	for (auto item : selected_used) {
		item->setSelected(true);
	}
	_list_used->scrollToTop();
	update_combination();
}

void PVGuiQt::PVAxesCombinationWidget::move_bottom_Slot()
{
	DisableDnD ddd(this);
	auto selected_used = ordered_selected(_list_used);
	if (selected_used.empty()) {
		return;
	}
	for (auto item_used : selected_used) {
		_list_used->takeItem(_list_used->row(item_used));
	}
	for (auto item_used : selected_used) {
		_list_used->addItem(item_used);
	}
	for (auto item : selected_used) {
		item->setSelected(true);
	}
	_list_used->scrollToBottom();
	update_combination();
}

void PVGuiQt::PVAxesCombinationWidget::gather_selected_Slot()
{
	DisableDnD ddd(this);
	auto selected_used = ordered_selected(_list_used);
	if (selected_used.count() < 2) {
		return;
	}
	int insert_row = _list_used->row(selected_used.front());
	for (auto item_used : selected_used) {
		_list_used->takeItem(_list_used->row(item_used));
	}
	for (auto item_used : selected_used) {
		_list_used->insertItem(insert_row++, item_used);
	}
	for (auto item : selected_used) {
		item->setSelected(true);
	}
	_list_used->scrollToItem(ordered_selected(_list_used).front());
	update_combination();
}

void PVGuiQt::PVAxesCombinationWidget::axis_remove_Slot()
{
	DisableDnD ddd(this);
	auto selected_used = _list_used->selectedItems();
	if (selected_used.empty()) {
		return;
	}
	for (auto item : selected_used) {
		delete item;
	}
	update_combination();
}

void PVGuiQt::PVAxesCombinationWidget::remove_duplicates_Slot()
{
	DisableDnD ddd(this);
	auto selected_used = ordered_selected(_list_used);
	std::vector<QListWidgetItem*> to_remove;
	auto find_all_to_remove = [&to_remove](int count, auto item_at) {
		std::unordered_set<PVCol> unique_cols;
		for (int i = 0; i < count; ++i) {
			QListWidgetItem* item = item_at(i);
			PVCol col = PVCol(item->data(Qt::UserRole).value<PVCol::value_type>());
			if (unique_cols.insert(col).second == false) {
				to_remove.push_back(item);
			}
		}
	};
	if (selected_used.count() > 1) {
		find_all_to_remove(selected_used.count(),
		                   [&selected_used](int i) { return selected_used[i]; });
	} else {
		find_all_to_remove(_list_used->count(), [this](int i) { return _list_used->item(i); });
	}
	for (auto item : to_remove) {
		delete item;
	}
	update_combination();
}

void PVGuiQt::PVAxesCombinationWidget::sort_Slot()
{
	DisableDnD ddd(this);
	auto selected_used = ordered_selected(_list_used);
	if (selected_used.count() < 2) {
		std::vector<QListWidgetItem*> sorted_selected;
		sorted_selected.reserve(_list_used->count());
		for (int i = 0; i < _list_used->count(); ++i) {
			sorted_selected.push_back(_list_used->item(i));
		}
		std::stable_sort(sorted_selected.begin(), sorted_selected.end(),
		                 [](auto a, auto b) { return a->text() < b->text(); });
		for (int i = _list_used->count(); i-- > 0;) {
			_list_used->takeItem(i);
		}
		for (auto & i : sorted_selected) {
			_list_used->addItem(i);
		}
		_list_used->selectAll();
	} else {
		auto sorted_selected = selected_used.toVector();
		std::stable_sort(sorted_selected.begin(), sorted_selected.end(),
		                 [](auto a, auto b) { return a->text() < b->text(); });
		std::vector<int> selected_pos(selected_used.size());
		std::transform(selected_used.begin(), selected_used.end(), selected_pos.begin(),
		               [this](auto item) { return _list_used->row(item); });
		std::sort(selected_pos.begin(), selected_pos.end());
		std::for_each(selected_pos.rbegin(), selected_pos.rend(),
		              [this](int pos) { _list_used->takeItem(pos); });
		for (int i = 0; i < sorted_selected.size(); ++i) {
			_list_used->insertItem(selected_pos[i], sorted_selected[i]);
			sorted_selected[i]->setSelected(true);
		}
	}
	update_combination();
}

void PVGuiQt::PVAxesCombinationWidget::update_orig_axes()
{
	_list_org->clear();
	for (PVCol col = PVCol(0); col < _axes_combination.get_nraw_axes_count(); ++col) {
		auto const& axis = _axes_combination.get_axis(col);
		auto item = new QListWidgetItem(axis.get_name());
		item->setData(Qt::UserRole, QVariant::fromValue<PVCol::value_type>(col));
		_list_org->addItem(item);
	}
	_label_axes_org->setText(QString::number(_list_org->count()) + " Axes");
}

void PVGuiQt::PVAxesCombinationWidget::update_used_axes()
{
	DisableDnD ddd(this);
	_list_used->clear();
	for (PVCol col : _axes_combination.get_combination()) {
		auto const& axis = _axes_combination.get_axis(col);
		auto item = new QListWidgetItem(axis.get_name());
		item->setData(Qt::UserRole, QVariant::fromValue<PVCol::value_type>(col));
		_list_used->addItem(item);
	}
	_label_axes_used->setText(QString::number(_list_used->count()) + " Axes");
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
	for (PVCombCol comb_col(0); comb_col < _axes_combination.get_axes_count(); comb_col++) {
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
	QList<PVCol> cols_rem = _view->get_parent<Squey::PVScaled>().get_singleton_columns_indexes();
	set_selection_from_cols(cols_rem);
}

void PVGuiQt::PVAxesCombinationWidget::update_combination()
{
	if (_list_used->count() < 2) {
		_axes_combination.reset_to_default();
		_label_axes_used->setText(QString::number(_axes_combination.get_axes_count()) + " Axes");
		return;
	}
	std::vector<PVCol> new_comb;
	new_comb.reserve(_list_used->count());
	for (int i = 0; i < _list_used->count(); ++i) {
		new_comb.emplace_back(_list_used->item(i)->data(Qt::UserRole).value<PVCol::value_type>());
	}
	_axes_combination.set_combination(new_comb);
	_label_axes_used->setText(QString::number(_list_used->count()) + " Axes");
}

void PVGuiQt::PVAxesCombinationWidget::enable_drop(bool enable)
{
	if (_dnd_enabled == enable) {
		return;
	}
	_dnd_enabled = enable;
	if (enable) {
		_connection_dnd_inserted =
		    connect(_list_used->model(), &QAbstractItemModel::rowsInserted, this,
		            [this](auto&...) { update_combination(); }, Qt::QueuedConnection);
		_connection_dnd_moved = connect(_list_used->model(), &QAbstractItemModel::rowsMoved,
		                                [this](auto&...) { update_combination(); });
		_connection_dnd_removed = connect(_list_used->model(), &QAbstractItemModel::rowsRemoved,
		                                  [this](auto&...) { update_combination(); });
	} else {
		disconnect(_connection_dnd_inserted);
		disconnect(_connection_dnd_moved);
		disconnect(_connection_dnd_removed);
	}
}

QList<QListWidgetItem*>
PVGuiQt::PVAxesCombinationWidget::ordered_selected(QListWidget* list_widget) const
{
	auto selected_used = list_widget->selectedItems();
	std::sort(selected_used.begin(), selected_used.end(), [list_widget](auto lh, auto rh) {
		return list_widget->row(lh) < list_widget->row(rh);
	});
	return selected_used;
}

void PVGuiQt::PVAxesCombinationWidget::closeEvent(QCloseEvent *event) {
	if (event->spontaneous()) {
		Q_EMIT closed();
	} else {
		QWidget::closeEvent(event);
	}
}