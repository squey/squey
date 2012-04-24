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
	_list = new PVSizeHintListWidget<QListView>();
	//_list->resize(_list->width(), 10) ;
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

void PVWidgets::PVAD2GEdgeEditor::set_cf(Picviz::PVView const& view_org, Picviz::PVView const& view_dst, Picviz::PVCombiningFunctionView& cf)
{
	if (_rff_list_model) {
		_rff_list_model->deleteLater();
	}
	_view_org = &view_org;
	_view_dst = &view_dst;
	_rff_list_model = new PVAD2GRFFListModel(view_org, view_dst, cf.get_first_tf()->get_rffs());
	_list->setModel(_rff_list_model);
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

	_rff_list_model->addRow(_list->selectionModel()->currentIndex(), new_rff);
	//_list->resize(_list->width(), _list->sizeHintForRow(0) * _rff_list_model->rowCount() + 10) ;

	emit rff_list_changed();
}

void PVWidgets::PVAD2GEdgeEditor::edit_function_Slot()
{
	if (!_rff_list_model) {
		return;
	}

	QModelIndex model_index = _list->selectionModel()->currentIndex();
	Picviz::PVSelRowFilteringFunction_p rff = ((Picviz::PVSelRowFilteringFunction*)model_index.data(Qt::UserRole).value<void*>())->shared_from_this();

	emit update_fonction_properties(*_view_org, *_view_dst, rff);
	//if (edit_rff(rff)) {
	QVariant var;
	var.setValue<void*>(rff.get());
	_rff_list_model->setData(model_index, var, Qt::UserRole);
	//}

	PVLOG_INFO("PVWidgets::PVAD2GEdgeEditor::edit_function_Slot() rff.get()=%x\n",rff.get());
}

void PVWidgets::PVAD2GEdgeEditor::update_item_Slot(const Picviz::PVSelRowFilteringFunction_p& rff)
{
	if (!_rff_list_model) {
		return;
	}

	QModelIndex model_index = _list->selectionModel()->currentIndex();
	QVariant var;
	var.setValue<void*>(rff.get());
	_rff_list_model->setData(model_index, var, Qt::UserRole);
}


void PVWidgets::PVAD2GEdgeEditor::remove_function_Slot()
{
	if (!_rff_list_model) {
		return;
	}

	QVariant var = _list->model()->data(_list->selectionModel()->currentIndex(), Qt::UserRole);
	Picviz::PVSelRowFilteringFunction* rff = (Picviz::PVSelRowFilteringFunction*) var.value<void*>();

	QMessageBox* box = new QMessageBox(QMessageBox::Question, tr("Confirm remove."), tr("Do you really want to remove row filter?"), QMessageBox::Yes | QMessageBox::No, this);
	if (box->exec() == QMessageBox::Yes) {
		_list->model()->removeRow(_list->selectionModel()->currentIndex().row());
		emit rff_removed_Signal(rff);
	}
}
