#include <picviz/PVCombiningFunctionView.h>
#include <picviz/PVSelRowFilteringFunction.h>
#include <picviz/PVTFViewRowFiltering.h>
#include <picviz/widgets/PVAD2GEdgeEditor.h>

#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QComboBox>
#include <QMessageBox>

PVWidgets::PVAD2GEdgeEditor::PVAD2GEdgeEditor(Picviz::PVView const& view_org, Picviz::PVView const& view_dst, Picviz::PVCombiningFunctionView& cf, QWidget* parent /*= 0*/) :
	QWidget(parent)/*,
	_tf(*cf.get_first_tf())*/,
	_rff_list_model(NULL),
	_view_org(view_org),
	_view_dst(view_dst)
{
	setWindowTitle("Edit combining function");

	// Widgets
	_list = new QListView();
	_rff_list_model = new PVAD2GRFFListModel(view_org, view_dst, cf.get_first_tf()->get_rffs());
	_list->setModel(_rff_list_model);
	_list->setDragDropMode(QAbstractItemView::InternalMove);
	_list->setDragDropOverwriteMode(true);
	_list->setMinimumWidth(400);
	QLabel* function_label = new QLabel("Function:");
	_function_combo = new QComboBox();
	init_combo_list_rffs();

	QPushButton* btn_add = new QPushButton(tr("+"));
	QPushButton* btn_edit = new QPushButton(tr("Edit"));
	QPushButton* btn_remove = new QPushButton(tr("Remove"));


	// Connections
	connect(btn_add, SIGNAL(clicked()), this, SLOT(add_function_Slot()));
	connect(btn_edit, SIGNAL(clicked()), this, SLOT(edit_function_Slot()));
	connect(btn_remove, SIGNAL(clicked()), this, SLOT(remove_function_Slot()));
	connect(_list, SIGNAL(doubleClicked(const QModelIndex &)), this, SLOT(edit_function_Slot()));

	// Layout
	QHBoxLayout* main_layout = new QHBoxLayout();
	QVBoxLayout* list_buttons_layout = new QVBoxLayout();
	QHBoxLayout* buttons_layout = new QHBoxLayout();
	buttons_layout->addWidget(function_label);
	buttons_layout->addWidget(_function_combo);
	buttons_layout->addWidget(btn_add);
	buttons_layout->addWidget(btn_edit);
	buttons_layout->addWidget(btn_remove);
	list_buttons_layout->addWidget(_list);
	list_buttons_layout->addLayout(buttons_layout);
	main_layout->addLayout(list_buttons_layout);
	setLayout(main_layout);
}

void PVWidgets::PVAD2GEdgeEditor::init_combo_list_rffs()
{
	LIB_CLASS(Picviz::PVSelRowFilteringFunction)::list_classes const& rffs = LIB_CLASS(Picviz::PVSelRowFilteringFunction)::get().get_list();
	LIB_CLASS(Picviz::PVSelRowFilteringFunction)::list_classes::const_iterator it;
	for (it = rffs.begin(); it != rffs.end(); it++) {
		_function_combo->addItem(it.value()->get_human_name(), QVariant(it.key()));
	}
}

void PVWidgets::PVAD2GEdgeEditor::update(Picviz::PVSelRowFilteringFunction_p& rff)
{
	emit update_fonction_properties(_view_org, _view_dst, rff);
}

void PVWidgets::PVAD2GEdgeEditor::add_function_Slot()
{
	QVariant var = _function_combo->itemData(_function_combo->currentIndex(), Qt::UserRole);
	Picviz::PVSelRowFilteringFunction_p new_rff = LIB_CLASS(Picviz::PVSelRowFilteringFunction)::get().get_class_by_name(var.toString());

	new_rff = new_rff->clone<Picviz::PVSelRowFilteringFunction>();

	_rff_list_model->addRow(_list->selectionModel()->currentIndex(), new_rff);
}

bool PVWidgets::PVAD2GEdgeEditor::edit_rff(Picviz::PVSelRowFilteringFunction_p& rff)
{
	//_functionPropertiesWidget->set_current_rff(*_view_org, *_view_dst, rff.get());
	/*PVAD2GFunctionPropertiesDialog* dlg = new PVAD2GFunctionPropertiesDialog(*_view_org, *_view_dst, *rff, this);
	if (dlg->exec() == QDialog::Accepted) {
		rff = dlg->get_rff();
		return true;
	}*/
	return false;
}

void PVWidgets::PVAD2GEdgeEditor::edit_function_Slot()
{
	QModelIndex model_index = _list->selectionModel()->currentIndex();
	Picviz::PVSelRowFilteringFunction_p rff = ((Picviz::PVSelRowFilteringFunction*)model_index.data(Qt::UserRole).value<void*>())->shared_from_this();

	if (edit_rff(rff)) {
		QVariant var;
		var.setValue<void*>(rff.get());
		_rff_list_model->setData(model_index, var, Qt::UserRole);
	}

}

void PVWidgets::PVAD2GEdgeEditor::remove_function_Slot()
{
	QMessageBox* box = new QMessageBox(QMessageBox::Question, tr("Confirm remove."), tr("Do you really want to remove row filter?"), QMessageBox::Yes | QMessageBox::No, this);
	if (box->exec() == QMessageBox::Yes) {
		_list->model()->removeRow(_list->selectionModel()->currentIndex().row());
	}
}
