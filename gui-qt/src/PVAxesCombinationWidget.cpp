#include <PVAxesCombinationWidget.h>
#include <picviz/PVAxesCombination.h>

PVInspector::PVAxesCombinationWidget::PVAxesCombinationWidget(Picviz::PVAxesCombination& axes_combination, QWidget* parent):
	QWidget(parent),
	_axes_combination(axes_combination)
{
	setupUi(this);

	_list_org->addItems(axes_combination.get_original_axes_names_list());
	update_used_axes();

	connect(_btn_axis_add, SIGNAL(clicked()), this, SLOT(axis_add_Slot()));
	connect(_btn_axis_up, SIGNAL(clicked()), this, SLOT(axis_up_Slot()));
	connect(_btn_axis_down, SIGNAL(clicked()), this, SLOT(axis_down_Slot()));
	connect(_btn_axis_remove, SIGNAL(clicked()), this, SLOT(axis_remove_Slot()));
}

void PVInspector::PVAxesCombinationWidget::axis_add_Slot()
{
	PVCol axis_id = get_original_axis_selected();
	QString axis_name = get_original_axis_selected_name();

	_list_used->addItem(axis_name);
	_axes_combination.axis_append(axis_id);

	_list_used->setCurrentRow(_list_used->count()-1);

	emit axes_count_changed();
	emit axes_combination_changed();
}

void PVInspector::PVAxesCombinationWidget::axis_up_Slot()
{
	if (!is_used_axis_selected()) {
		return;
	}

	PVCol axis_id = get_used_axis_selected();	
	if (axis_id == 0) {
		return;
	}

	_axes_combination.move_axis_left_one_position(axis_id);
	update_used_axes();
	_list_used->setCurrentRow(axis_id-1);

	emit axes_combination_changed();
}

void PVInspector::PVAxesCombinationWidget::axis_down_Slot()
{
	if (!is_used_axis_selected()) {
		return;
	}

	PVCol axis_id = get_used_axis_selected();
	if (axis_id == _list_used->count() - 1) {
		return;
	}

	_axes_combination.move_axis_right_one_position(axis_id);
	update_used_axes();
	_list_used->setCurrentRow(axis_id+1);

	emit axes_combination_changed();
}

void PVInspector::PVAxesCombinationWidget::axis_remove_Slot()
{
	if (!is_used_axis_selected()) {
		return;
	}

	PVCol axis_id = get_used_axis_selected();	
	_axes_combination.remove_axis(axis_id);
	update_used_axes();
	_list_used->setCurrentRow(picviz_min(axis_id, _list_used->count()-1));

	emit axes_count_changed();
	emit axes_combination_changed();
}

PVCol PVInspector::PVAxesCombinationWidget::get_original_axis_selected()
{
	return _list_org->currentRow();
}

QString PVInspector::PVAxesCombinationWidget::get_original_axis_selected_name()
{
	return _list_org->currentItem()->text();
}

PVCol PVInspector::PVAxesCombinationWidget::get_used_axis_selected()
{
	return _list_used->currentRow();
}

void PVInspector::PVAxesCombinationWidget::update_used_axes()
{
	_list_used->clear();
	_list_used->addItems(_axes_combination.get_axes_names_list());
}

bool PVInspector::PVAxesCombinationWidget::is_used_axis_selected()
{
	return _list_used->selectedItems().size() > 0;
}

void PVInspector::PVAxesCombinationWidget::save_current_combination()
{
	_saved_combination = _axes_combination;
}

void PVInspector::PVAxesCombinationWidget::restore_saved_combination()
{
	bool count_changed = (_axes_combination.get_axes_count() != _saved_combination.get_axes_count());
	_axes_combination = _saved_combination;
	if (count_changed) {
		emit axes_count_changed();
	}
	emit axes_combination_changed();
}
