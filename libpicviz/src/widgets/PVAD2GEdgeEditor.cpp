#include <picviz/PVCombiningFunctionView.h>
#include <picviz/PVSelRowFilteringFunction.h>
#include <picviz/PVTFViewRowFiltering.h>
#include <picviz/widgets/PVAD2GEdgeEditor.h>
#include <picviz/widgets/PVAD2GFunctionPropertiesDialog.h>

#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QMessageBox>

PVWidgets::PVAD2GEdgeEditor::PVAD2GEdgeEditor(Picviz::PVView const& view_org, Picviz::PVView const& view_dst, Picviz::PVCombiningFunctionView& cf, QWidget* parent /*= 0*/) :
	QDialog(parent),
	_tf(*cf.get_first_tf()),
	_view_org(view_org),
	_view_dst(view_dst)
{
	setWindowTitle("Edit combining function");

	// Widgets
	_list = new QListView();
	QPushButton* btn_add = new QPushButton(tr("Add"));
	QPushButton* btn_edit = new QPushButton(tr("Edit"));
	QPushButton* btn_remove = new QPushButton(tr("Remove"));

	// Model
	_rff_list_model = new PVAD2GRFFListModel(view_org, view_dst, _tf.get_rffs());
	_list->setModel(_rff_list_model);
	_list->setDragDropMode(QAbstractItemView::InternalMove);
	_list->setDragDropOverwriteMode(true);

	// Connections
	connect(btn_add, SIGNAL(clicked()), this, SLOT(add_function_Slot()));
	connect(btn_edit, SIGNAL(clicked()), this, SLOT(edit_function_Slot()));
	connect(btn_remove, SIGNAL(clicked()), this, SLOT(remove_function_Slot()));
	connect(_list, SIGNAL(doubleClicked(const QModelIndex &)), this, SLOT(edit_function_Slot()));

	// Layout
	QHBoxLayout* main_layout = new QHBoxLayout();
	QVBoxLayout* buttons_layout = new QVBoxLayout();
	buttons_layout->addWidget(btn_add);
	buttons_layout->addWidget(btn_edit);
	buttons_layout->addWidget(btn_remove);
	main_layout->addWidget(_list);
	main_layout->addLayout(buttons_layout);
	setLayout(main_layout);
}

Picviz::PVSelRowFilteringFunction_p PVWidgets::PVAD2GEdgeEditor::get_default_rff()
{
	// Get first RFF plugin in the list
	LIB_CLASS(Picviz::PVSelRowFilteringFunction)::list_classes const& classes = LIB_CLASS(Picviz::PVSelRowFilteringFunction)::get().get_list();
	if (classes.size() == 0) {
		return Picviz::PVSelRowFilteringFunction_p();
	}

	return classes.begin().value();
}


bool PVWidgets::PVAD2GEdgeEditor::edit_rff(Picviz::PVSelRowFilteringFunction_p& rff)
{
	PVAD2GFunctionPropertiesDialog* dlg = new PVAD2GFunctionPropertiesDialog(_view_org, _view_dst, *rff, this);
	if (dlg->exec() == QDialog::Accepted) {
		rff = dlg->get_rff();
		return true;
	}
	return false;
}

void PVWidgets::PVAD2GEdgeEditor::add_function_Slot()
{
	Picviz::PVSelRowFilteringFunction_p new_rff = get_default_rff();
	if (!new_rff) {
		return;
	}

	new_rff = new_rff->clone<Picviz::PVSelRowFilteringFunction>();

	if (edit_rff(new_rff)) {
		_rff_list_model->addRow(_list->selectionModel()->currentIndex(), new_rff);
		_tf.set_rffs(_rff_list_model->getRFFList());

//		QModelIndex model_index = _list->selectionModel()->currentIndex();
//		_list->model()->insertRow(model_index.row());
//		QVariant var;
//		var.setValue<void*>(new_rff.get());
//		_list->model()->setData(model_index, var, Qt::UserRole);
		//_tf.push_rff(new_rff);
	}
}

void PVWidgets::PVAD2GEdgeEditor::edit_function_Slot()
{
	QModelIndex model_index = _list->selectionModel()->currentIndex();
	Picviz::PVSelRowFilteringFunction_p rff((Picviz::PVSelRowFilteringFunction*)model_index.data(Qt::UserRole).value<void*>());

	edit_rff(rff);
}

void PVWidgets::PVAD2GEdgeEditor::remove_function_Slot()
{
	QMessageBox* box = new QMessageBox(QMessageBox::Question, tr("Confirm remove."), tr("Do you really want to remove row filter?"), QMessageBox::Yes | QMessageBox::No, this);
	if (box->exec() == QMessageBox::Yes) {
		_list->model()->removeRow(_list->selectionModel()->currentIndex().row());
		_tf.set_rffs(_rff_list_model->getRFFList());
		//_tf.remove_rff(_list->selectionModel()->currentIndex().row());
		//_list->model()->reset();
	}
}
