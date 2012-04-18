#include <picviz/PVCombiningFunctionView.h>
#include <picviz/PVSelRowFilteringFunction.h>
#include <picviz/PVTFViewRowFiltering.h>
#include <picviz/widgets/PVAD2GEdgeEditor.h>
#include <picviz/widgets/PVAD2GFunctionPropertiesDialog.h>

#include <QListWidget>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>

PVWidgets::PVAD2GEdgeEditor::PVAD2GEdgeEditor(Picviz::PVView const& view_org, Picviz::PVView const& view_dst, Picviz::PVCombiningFunctionView& cf, QWidget* parent /*= 0*/) :
	QDialog(parent),
	_tf(*cf.get_first_tf()),
	_view_org(view_org),
	_view_dst(view_dst)
{
	setWindowTitle("Edit combining function");

	// Widgets
	QListWidget* list = new QListWidget();
	QPushButton* btn_add = new QPushButton(tr("Add"));
	QPushButton* btn_edit = new QPushButton(tr("Edit"));
	QPushButton* btn_remove = new QPushButton(tr("Remove"));

	// Connections
	connect(btn_add, SIGNAL(clicked()), this, SLOT(add_function_Slot()));
	connect(btn_edit, SIGNAL(clicked()), this, SLOT(edit_function_Slot()));
	connect(btn_remove, SIGNAL(clicked()), this, SLOT(remove_function_Slot()));

	// Layout
	QHBoxLayout* main_layout = new QHBoxLayout();
	QVBoxLayout* buttons_layout = new QVBoxLayout();
	buttons_layout->addWidget(btn_add);
	buttons_layout->addWidget(btn_edit);
	buttons_layout->addWidget(btn_remove);
	main_layout->addWidget(list);
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

void PVWidgets::PVAD2GEdgeEditor::add_function_Slot()
{
	Picviz::PVSelRowFilteringFunction_p new_rff = get_default_rff();
	if (!new_rff) {
		return;
	}

	new_rff = new_rff->clone<Picviz::PVSelRowFilteringFunction>();

	if (edit_rff(new_rff)) {
		_tf.push_rff(new_rff);
	}
}

void PVWidgets::PVAD2GEdgeEditor::edit_function_Slot()
{
}

void PVWidgets::PVAD2GEdgeEditor::remove_function_Slot()
{
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
