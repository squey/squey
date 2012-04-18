#include <picviz/widgets/PVAD2GFunctionPropertiesDialog.h>
#include <picviz/widgets/PVAD2GFunctionPropertiesWidget.h>

#include <QDialogButtonBox>
#include <QVBoxLayout>

PVWidgets::PVAD2GFunctionPropertiesDialog::PVAD2GFunctionPropertiesDialog(Picviz::PVView const& view_org, Picviz::PVView const& view_dst, Picviz::PVSelRowFilteringFunction const& rff, QWidget* parent):
	QDialog(parent)
{
	// Widgets
	_props = new PVAD2GFunctionPropertiesWidget(view_org, view_dst, rff, this);
	QDialogButtonBox* btns = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

	// Connections
	connect(btns, SIGNAL(accepted()), this, SLOT(accept()));
	connect(btns, SIGNAL(rejected()), this, SLOT(reject()));

	// Layout
	QVBoxLayout* main_layout = new QVBoxLayout();
	main_layout->addWidget(_props);
	main_layout->addWidget(btns);

	setLayout(main_layout);
}
