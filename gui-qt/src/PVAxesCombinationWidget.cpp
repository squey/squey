#include <PVAxesCombinationWidget.h>
#include <picviz/PVAxesCombination.h>

#include <QDialogButtonBox>

PVInspector::PVAxesCombinationWidget::PVAxesCombinationWidget(Picviz::PVAxesCombination& axes_combination, QWidget* parent):
	QWidget(parent),
	_axes_combination(axes_combination)
{
	setupUi(this);

	update_all();

	_move_dlg = new PVMoveToDlg(this);

	connect(_btn_axis_add, SIGNAL(clicked()), this, SLOT(axis_add_Slot()));
	connect(_btn_axis_up, SIGNAL(clicked()), this, SLOT(axis_up_Slot()));
	connect(_btn_axis_down, SIGNAL(clicked()), this, SLOT(axis_down_Slot()));
	connect(_btn_axis_move, SIGNAL(clicked()), this, SLOT(axis_move_Slot()));
	connect(_btn_axis_remove, SIGNAL(clicked()), this, SLOT(axis_remove_Slot()));
	connect(_btn_reset, SIGNAL(clicked()), this, SLOT(reset_comb_Slot()));
}

void PVInspector::PVAxesCombinationWidget::axis_add_Slot()
{
	if (!is_original_axis_selected()) {
		return;
	}

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

void PVInspector::PVAxesCombinationWidget::axis_move_Slot()
{
	if (!is_used_axis_selected()) {
		return;
	}

	_move_dlg->update_axes();
	if (_move_dlg->exec() != QDialog::Accepted) {
		return;
	}

	PVCol org = get_used_axis_selected();
	PVCol dest = _move_dlg->get_dest_col(org);
	if (!_axes_combination.move_axis_to_new_position(org, dest)) {
		return;
	}

	update_used_axes();
	_list_used->setCurrentRow(dest);
	emit axes_combination_changed();
}

void PVInspector::PVAxesCombinationWidget::axis_remove_Slot()
{
	if (!is_used_axis_selected()) {
		return;
	}

	// We need a minimum of 2 axes !
	if (_list_used->count() <= 2) {
		return;
	}

	PVCol axis_id = get_used_axis_selected();	
	_axes_combination.remove_axis(axis_id);
	update_used_axes();
	_list_used->setCurrentRow(picviz_min(axis_id, _list_used->count()-1));

	emit axes_count_changed();
	emit axes_combination_changed();
}

void PVInspector::PVAxesCombinationWidget::reset_comb_Slot()
{
	PVCol nold_axes = _axes_combination.get_axes_count();
	_axes_combination.reset_to_default();
	
	update_used_axes();

	if (nold_axes != _axes_combination.get_axes_count()) {
		emit axes_count_changed();
	}
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

void PVInspector::PVAxesCombinationWidget::update_orig_axes()
{
	_list_org->clear();
	_list_org->addItems(_axes_combination.get_original_axes_names_list());
}

void PVInspector::PVAxesCombinationWidget::update_used_axes()
{
	_list_used->clear();
	_list_used->addItems(_axes_combination.get_axes_names_list());
}

void PVInspector::PVAxesCombinationWidget::update_all()
{
	update_orig_axes();
	update_used_axes();
}

bool PVInspector::PVAxesCombinationWidget::is_used_axis_selected()
{
	return _list_used->selectedItems().size() > 0;
}

bool PVInspector::PVAxesCombinationWidget::is_original_axis_selected()
{
	return _list_org->selectedItems().size() > 0;
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

// PVMoveToDlg implementation
PVInspector::PVAxesCombinationWidget::PVMoveToDlg::PVMoveToDlg(PVAxesCombinationWidget* parent):
	QDialog(parent),
	_parent(parent)
{
	setWindowTitle(tr("Move to..."));

	QVBoxLayout* main_layout = new QVBoxLayout();
	QHBoxLayout* combo_layout = new QHBoxLayout();

	_after_combo = new QComboBox();
	_after_combo->addItem(tr("Before"));
	_after_combo->addItem(tr("After"));

	_axes_combo = new QComboBox();
	_axes_combo->addItems(parent->_axes_combination.get_axes_names_list());

	combo_layout->addWidget(_after_combo);
	combo_layout->addWidget(_axes_combo);
	main_layout->addLayout(combo_layout);

	QDialogButtonBox* btns = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	connect(btns, SIGNAL(accepted()), this, SLOT(accept()));
	connect(btns, SIGNAL(rejected()), this, SLOT(reject()));
	main_layout->addWidget(btns);

	setLayout(main_layout);
}

PVCol PVInspector::PVAxesCombinationWidget::PVMoveToDlg::get_dest_col(PVCol org)
{
	bool after = _after_combo->currentIndex() == 1;
	PVCol pos = _axes_combo->currentIndex();
	if (after) {
		pos++;
	}

	if (pos > org) {
		pos--;
	}

	return pos;
}

void PVInspector::PVAxesCombinationWidget::PVMoveToDlg::update_axes()
{
	QString _old_sel = _axes_combo->currentText();
	_axes_combo->clear();
	_axes_combo->addItems(_parent->_axes_combination.get_axes_names_list());
	int old_sel_idx = _axes_combo->findText(_old_sel);
	if (old_sel_idx >= 0) {
		_axes_combo->setCurrentIndex(old_sel_idx);
	}
}
