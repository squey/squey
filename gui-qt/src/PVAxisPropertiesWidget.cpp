//! \file PVAxisPropertiesWidget.cpp
//! $Id: PVAxisPropertiesWidget.cpp 3236 2011-07-04 12:40:17Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtGui>

#include <pvcore/general.h>

#include <PVMainWindow.h>

#include <PVAxisPropertiesWidget.h>

/******************************************************************************
 *
 * PVInspector::PVAxisPropertiesWidget::PVAxisPropertiesWidget
 *
 *****************************************************************************/
PVInspector::PVAxisPropertiesWidget::PVAxisPropertiesWidget(PVMainWindow *mw) : QDialog(mw)
{


	PVLOG_DEBUG("PVInspector::PVAxisPropertiesWidget::%s\n", __FUNCTION__);

	// Setting the direct link to PVMainWindow
	main_window = mw;

	apply_button = new QPushButton("Apply");
	axes_list = new QComboBox();
	axes_names_list = QStringList();
	axis_name = new QLineEdit();
	axis_name_label = new QLabel("Axis name");
	cancel_button = new QPushButton("Cancel");
	done_button = new QPushButton("Done");
	main_layout = new QGridLayout();

	done_button->setDefault(true);

}

/******************************************************************************
 *
 * PVInspector::PVAxisPropertiesWidget::~PVAxisPropertiesWidget
 *
 *****************************************************************************/
PVInspector::PVAxisPropertiesWidget::~PVAxisPropertiesWidget()
{
	apply_button->deleteLater();
	axes_list->deleteLater();
	axis_name->deleteLater();
	axis_name_label->deleteLater();
	cancel_button->deleteLater();
	done_button->deleteLater();
	main_layout->deleteLater();
}

/******************************************************************************
 *
 * PVInspector::PVAxisPropertiesWidget::apply_slot
 *
 *****************************************************************************/
void PVInspector::PVAxisPropertiesWidget::apply_slot()
{
	PVLOG_INFO("%s \n       %s\n",__FILE__,__FUNCTION__);

	main_window->current_tab->get_lib_view()->set_axis_name(axes_list->currentIndex(), axis_name->text());

	refresh_widget(axes_list->currentIndex());
}

/******************************************************************************
 *
 * PVInspector::PVAxisPropertiesWidget::cancel_slot
 *
 *****************************************************************************/
void PVInspector::PVAxisPropertiesWidget::cancel_slot(){
	PVLOG_INFO("%s \n       %s\n",__FILE__,__FUNCTION__);
	//get current list index
	int idx = axes_list->currentIndex();
	//reload
	refresh_widget(idx);
}

/******************************************************************************
 *
 * PVInspector::PVAxisPropertiesWidget::create
 *
 *****************************************************************************/
void PVInspector::PVAxisPropertiesWidget::create()
{
	PVLOG_INFO("%s \n       %s\n",__FILE__,__FUNCTION__);

	axes_list->clear();
	axes_names_list = main_window->current_tab->get_lib_view()->get_axes_names_list();
	axes_list->addItems(axes_names_list);

	axis_name->setText(axes_names_list[0]);

	main_layout->addWidget(axes_list, 0, 0, 1, 2);
	
	main_layout->addWidget(axis_name_label, 1, 0);
	main_layout->addWidget(axis_name, 1, 1);

	main_layout->addWidget(apply_button, 2, 0);
	main_layout->addWidget(cancel_button, 2, 1);
	main_layout->addWidget(done_button, 2, 2);

	this->setLayout(main_layout);

	//buttons
	connect(apply_button, 	SIGNAL(clicked()), 	this, 	SLOT(apply_slot())	);
	connect(cancel_button, 	SIGNAL(clicked()), 	this, 	SLOT(cancel_slot())	);
	connect(done_button, 	SIGNAL(clicked()), 	this, 	SLOT(apply_slot())	);
	connect(done_button, 	SIGNAL(clicked()), 	this, 	SLOT(accept())		);
	
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
	PVLOG_INFO("%s \n       %s\n",__FILE__,__FUNCTION__);
	refresh_widget(0);
}

/******************************************************************************
 *
 * PVInspector::PVAxisPropertiesWidget::refresh_widget
 *
 *****************************************************************************/
void PVInspector::PVAxisPropertiesWidget::refresh_widget(int index)
{
	PVLOG_INFO("%s \n       %s\n",__FILE__,__FUNCTION__);
	axes_list->clear();
	axes_names_list = main_window->current_tab->get_lib_view()->get_axes_names_list();
	axes_list->addItems(axes_names_list);
	axes_list->setCurrentIndex(index);

	axis_name->setText(axes_list->currentText());
}



/* For information :
	float absciss;
	 VCol column_index; Needed in PVAxesCombination, we shall remove it 
	Picviz::PVColor titlecolor;
	Picviz::PVColor color;
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


