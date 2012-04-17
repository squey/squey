#include <picviz/widgets/PVAD2GEdgeEditor.h>


Picviz::PVAD2GEdgeEditor::PVAD2GEdgeEditor(QWidget* parent /*= 0*/) :
	QDialog(parent)
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

void Picviz::PVAD2GEdgeEditor::add_function_Slot()
{
	PVAD2GFunctionProperties* function_properties = new PVAD2GFunctionProperties();
	function_properties->exec();
}

void Picviz::PVAD2GEdgeEditor::edit_function_Slot()
{
}

void Picviz::PVAD2GEdgeEditor::remove_function_Slot()
{
}

Picviz::PVAD2GFunctionProperties::PVAD2GFunctionProperties(QWidget* parent /*= 0*/) :
	QDialog(parent)
{
	setWindowTitle("Function properties");

	// Widgets
	QLabel* function_label = new QLabel(tr("Function: "));
	QComboBox* function_combo = new QComboBox();
	QGroupBox* src_view_box = new QGroupBox(tr("Properties for original view"));
	QGroupBox* dst_view_box = new QGroupBox(tr("Properties for destination view"));

	// Layout
	QVBoxLayout* main_layout = new QVBoxLayout();
	QHBoxLayout* function_type_layout = new QHBoxLayout();
	function_type_layout->addWidget(function_label);
	function_type_layout->addWidget(function_combo);
	main_layout->addLayout(function_type_layout);
	main_layout->addWidget(src_view_box);
	main_layout->addWidget(dst_view_box);

	setLayout(main_layout);
}
