#include <pvkernel/core/PVClassLibrary.h>
#include <picviz/PVCombiningFunctionView.h>
#include <picviz/PVSelRowFilteringFunction.h>
#include <picviz/PVTFViewRowFiltering.h>
#include <picviz/widgets/PVAD2GEdgeEditor.h>
#include <picviz/widgets/PVCombinOpDelegate.h>

#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QComboBox>
#include <QMessageBox>
#include <QHeaderView>

PVWidgets::PVAD2GEdgeEditor::PVAD2GEdgeEditor(QWidget* parent /*= 0*/) :
	QWidget(parent),
	_rff_list_model(NULL),
	_view_org(NULL),
	_view_dst(NULL)
{
	init();
}

PVWidgets::PVAD2GEdgeEditor::PVAD2GEdgeEditor(Picviz::PVView const& view_org, Picviz::PVView const& view_dst, Picviz::PVCombiningFunctionView& cf, QWidget* parent /*= 0*/):
	QWidget(parent),
	_rff_list_model(NULL),
	_view_org(NULL),
	_view_dst(NULL)
{
	init();
	set_cf(view_org, view_dst, cf);
}

void PVWidgets::PVAD2GEdgeEditor::init()
{
	setWindowTitle("Edit combining function");

	// Widgets
	_list = new PVSizeHintListWidget<QTableView>();
	_combin_op_delegate = new PVWidgets::PVCombinOpDelegate(this);

	// initializing the QTableView
	_list->setItemDelegateForColumn(0, _combin_op_delegate);
	_list->setShowGrid(false);
	_list->horizontalHeader()->hide();
	_list->verticalHeader()->hide();
	_list->setMinimumWidth(_list->horizontalHeader()->length());
	_list->horizontalHeader()->setStretchLastSection(true);

	_list->setSelectionMode(QAbstractItemView::SingleSelection);
	_list->setDragDropMode(QAbstractItemView::InternalMove);
	_list->setDragDropOverwriteMode(true);
	_list->setMinimumWidth(400);
	QLabel* function_label = new QLabel("Function:");
	_function_combo = new QComboBox();
	init_combo_list_rffs();

	QPushButton* btn_add = new QPushButton(tr("Add"));
	_btn_up = new QPushButton(tr("Up"));
	_btn_down = new QPushButton(tr("Down"));
	_btn_remove = new QPushButton(tr("Remove"));

	// Connections
	connect(btn_add, SIGNAL(clicked()), this, SLOT(add_function_Slot()));
	connect(_btn_remove, SIGNAL(clicked()), this, SLOT(remove_function_Slot()));
	connect(_btn_up, SIGNAL(clicked()), this, SLOT(move_function_up_Slot()));
	connect(_btn_down, SIGNAL(clicked()), this, SLOT(move_function_down_Slot()));
	connect(_list, SIGNAL(clicked(const QModelIndex &)), this, SLOT(edit_function_Slot()));

	// Layout
	QHBoxLayout* main_layout = new QHBoxLayout();
	QVBoxLayout* list_buttons_layout = new QVBoxLayout();
	QHBoxLayout* buttons_layout = new QHBoxLayout();
	buttons_layout->addWidget(function_label);
	buttons_layout->addWidget(_function_combo);
	buttons_layout->addWidget(btn_add);
	buttons_layout->addWidget(_btn_up);
	buttons_layout->addWidget(_btn_down);
	buttons_layout->addWidget(_btn_remove);
	list_buttons_layout->addWidget(_list);
	list_buttons_layout->addLayout(buttons_layout);
	main_layout->addLayout(list_buttons_layout);
	setLayout(main_layout);
}

void PVWidgets::PVAD2GEdgeEditor::set_no_cf()
{
	if (_rff_list_model) {
		_list->selectionModel()->clearSelection();
		_list->setModel(0);
		_rff_list_model->deleteLater();
		_rff_list_model = 0;
	}
}

void PVWidgets::PVAD2GEdgeEditor::set_cf(Picviz::PVView const& view_org, Picviz::PVView const& view_dst, Picviz::PVCombiningFunctionView& cf)
{
	if (_rff_list_model) {
		_rff_list_model->deleteLater();
	}
	_view_org = &view_org;
	_view_dst = &view_dst;
	_rff_list_model = new PVAD2GRFFListModel(view_org, view_dst, cf.get_first_tf()->get_rffs());
	_list->setModel(_rff_list_model);
	connect(_list->selectionModel(), SIGNAL(currentChanged(QModelIndex, QModelIndex)), this, SLOT(update_buttons_status()));
}

void PVWidgets::PVAD2GEdgeEditor::init_combo_list_rffs()
{
	LIB_CLASS(Picviz::PVSelRowFilteringFunction)::list_classes const& rffs = LIB_CLASS(Picviz::PVSelRowFilteringFunction)::get().get_list();
	LIB_CLASS(Picviz::PVSelRowFilteringFunction)::list_classes::const_iterator it;
	for (it = rffs.begin(); it != rffs.end(); it++) {
		_function_combo->addItem(it.value()->get_human_name(), QVariant(it.key()));
	}
}

void PVWidgets::PVAD2GEdgeEditor::add_function_Slot()
{
	if (!_rff_list_model) {
		return;
	}

	QVariant var = _function_combo->itemData(_function_combo->currentIndex(), Qt::UserRole);
	Picviz::PVSelRowFilteringFunction_p new_rff = LIB_CLASS(Picviz::PVSelRowFilteringFunction)::get().get_class_by_name(var.toString());

	new_rff = new_rff->clone<Picviz::PVSelRowFilteringFunction>();

	// make sure the new RFF is added in the correct column (it is forced to 1)
	QModelIndex current_index = _list->selectionModel()->currentIndex();
	QModelIndex model_index = current_index.sibling(current_index.row(), 1);
	_rff_list_model->addRow(model_index, new_rff);

	emit rff_list_changed();

	update_buttons_status();

	// Update selection
	_list->selectionModel()->clearSelection();
	_list->selectionModel()->setCurrentIndex(model_index, QItemSelectionModel::Select);
}


void PVWidgets::PVAD2GEdgeEditor::edit_function_Slot()
{
	if (!_rff_list_model) {
		return;
	}

	QModelIndex model_index = _list->selectionModel()->currentIndex();

	if (model_index.column() != 1) {
		return;
	}

	Picviz::PVSelRowFilteringFunction_p rff = ((Picviz::PVSelRowFilteringFunction*)model_index.data(Qt::UserRole).value<void*>())->shared_from_this();

	_cur_edited_rff_index = model_index;

	emit update_fonction_properties(*_view_org, *_view_dst, rff);
}

void PVWidgets::PVAD2GEdgeEditor::update_buttons_status()
{
	QModelIndex model_index = _list->selectionModel()->currentIndex();
	_btn_up->setEnabled(model_index.row() > 0);
	_btn_down->setEnabled(model_index.row() < _rff_list_model->rowCount()-1 && _rff_list_model->rowCount() > 1);
	_btn_remove->setEnabled(_rff_list_model->rowCount() > 0);
}

void PVWidgets::PVAD2GEdgeEditor::move_function(bool up)
{
	QModelIndexList items = _list->selectionModel()->selectedIndexes();
	if (items.count() == 1) {
		QModelIndex index = items.at(0);
		int row = index.row();
		_rff_list_model->move_index(index, up);

		// Update selection
		_list->selectionModel()->clearSelection();
		_list->selectionModel()->setCurrentIndex(_rff_list_model->index(row + (up ? -1 : 1), 1), QItemSelectionModel::Select);
	}
}

void PVWidgets::PVAD2GEdgeEditor::update_item_Slot(const Picviz::PVSelRowFilteringFunction_p& rff)
{
	if (!_rff_list_model) {
		return;
	}

	QVariant var;
	var.setValue<void*>(rff.get());
	_rff_list_model->setData(_cur_edited_rff_index, var, Qt::UserRole);
}


void PVWidgets::PVAD2GEdgeEditor::remove_function_Slot()
{
	if (!_rff_list_model) {
		return;
	}

	QModelIndex idx_rem = _list->selectionModel()->currentIndex();
	if (!idx_rem.isValid()) {
		return;
	}

	QVariant var = _list->model()->data(idx_rem, Qt::UserRole);

	QMessageBox* box = new QMessageBox(QMessageBox::Question, tr("Confirm remove."), tr("Do you really want to remove row filter?"), QMessageBox::Yes | QMessageBox::No, this);
	if (box->exec() == QMessageBox::Yes) {
		if (idx_rem == _cur_edited_rff_index) {
			emit cur_rff_removed();
		}
		_list->model()->removeRow(_list->selectionModel()->currentIndex().row());
		update_buttons_status();
	}
}
