//! \file PVAxisPropertiesWidget.cpp
//! $Id: PVAxisPropertiesWidget.cpp 3236 2011-07-04 12:40:17Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvkernel/core/general.h>

#include <PVAxisPropertiesWidget.h>
#include <PVMainWindow.h>
#include <PVTabSplitter.h>

/******************************************************************************
 *
 * PVInspector::PVAxisPropertiesWidget::PVAxisPropertiesWidget
 *
 *****************************************************************************/
PVInspector::PVAxisPropertiesWidget::PVAxisPropertiesWidget(Picviz::PVView_sp view, PVTabSplitter* tab_, PVMainWindow *mw):
	QDialog(mw),
	_view(view)
{
	main_window = mw;

	PVLOG_DEBUG("PVInspector::PVAxisPropertiesWidget::%s\n", __FUNCTION__);

	// Setting the direct link to PVMainWindow
	tab = tab_;

	axes_list = new QComboBox();
	axes_names_list = QStringList();
	axis_name = new QLineEdit();
	axis_name_label = new QLabel("Axis name");
	box_buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel | QDialogButtonBox::Apply);
	grid_layout = new QGridLayout();
	main_layout = new QVBoxLayout();

	setWindowTitle(tr("Edit axes..."));

	create();
}

/******************************************************************************
 *
 * PVInspector::PVAxisPropertiesWidget::~PVAxisPropertiesWidget
 *
 *****************************************************************************/
PVInspector::PVAxisPropertiesWidget::~PVAxisPropertiesWidget()
{
}

/******************************************************************************
 *
 * PVInspector::PVAxisPropertiesWidget::apply_slot
 *
 *****************************************************************************/
void PVInspector::PVAxisPropertiesWidget::apply_slot()
{
	PVLOG_INFO("%s \n       %s\n",__FILE__,__FUNCTION__);

	_view->set_axis_name(axes_list->currentIndex(), axis_name->text());
	refresh_widget(axes_list->currentIndex());
}

/******************************************************************************
 *
 * PVInspector::PVAxisPropertiesWidget::cancel_slot
 *
 *****************************************************************************/
void PVInspector::PVAxisPropertiesWidget::cancel_slot()
{
	reject();
}

/******************************************************************************
 *
 * PVInspector::PVAxisPropertiesWidget::create
 *
 *****************************************************************************/
void PVInspector::PVAxisPropertiesWidget::create()
{
	axes_list->clear();
	axes_names_list = _view->get_axes_names_list();
	axes_list->addItems(axes_names_list);

	axis_name->setText(axes_names_list[0]);

	grid_layout->addWidget(axes_list, 0, 0, 1, 2);
	
	grid_layout->addWidget(axis_name_label, 1, 0);
	grid_layout->addWidget(axis_name, 1, 1);

	main_layout->addLayout(grid_layout);
	main_layout->addWidget(box_buttons);

	setLayout(main_layout);

	//buttons
	connect(box_buttons,    SIGNAL(accepted()),   this,   SLOT(apply_slot())  );
	connect(box_buttons,    SIGNAL(accepted()),   this,   SLOT(accept())      );
	connect(box_buttons,    SIGNAL(rejected()), this,   SLOT(cancel_slot())   );
	connect(box_buttons->button(QDialogButtonBox::Apply), SIGNAL(clicked()), this, SLOT(apply_slot()));
	
	//combobox list 
// 	connect(axes_list, 	SIGNAL(currentIndexChanged(int)), this, SLOT(refresh_widget(int)));
	connect(axes_list, 	SIGNAL(activated(int)), this, 	SLOT(refresh_widget(int)));
}

/******************************************************************************
 *
 * PVInspector::PVAxisPropertiesWidget::refresh_widget
 *
 *****************************************************************************/
void PVInspector::PVAxisPropertiesWidget::refresh_widget()
{	
	refresh_widget(0);
}

/******************************************************************************
 *
 * PVInspector::PVAxisPropertiesWidget::refresh_widget
 *
 *****************************************************************************/
void PVInspector::PVAxisPropertiesWidget::refresh_widget(int index)
{
	axes_list->clear();
	axes_names_list = _view->get_axes_names_list();
	axes_list->addItems(axes_names_list);
	axes_list->setCurrentIndex(index);

	axis_name->setText(axes_list->currentText());
}


/* For information :
	float absciss;
	 VCol column_index; Needed in PVAxesCombination, we shall remove it 
	PVCore::PVColor titlecolor;
	PVCore::PVColor color;
	bool is_expandable;
	bool is_expanded;
	bool is_key;
	QString name;
	QString type;
	QString modemapping;
	QString modeplotting;
	float thickness;

is Picviz:PVAxis
*/


