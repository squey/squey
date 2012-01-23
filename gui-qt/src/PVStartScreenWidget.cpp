//! \file PVStartScreenWidget.cpp
//! $Id: PVStartScreenWidget.cpp 3196 2011-06-23 16:24:50Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2012
//! Copyright (C) Philippe Saadé 2009-2012
//! Copyright (C) Picviz Labs 2012

#include <QtGui>

#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>



#include <pvkernel/core/general.h>
#include <picviz/PVView.h>

#include <PVMainWindow.h>
#include <PVTabSplitter.h>

#include <PVStartScreenWidget.h>

/******************************************************************************
 *
 * PVInspector::PVStartScreenWidget::PVStartScreenWidget
 *
 *****************************************************************************/
PVInspector::PVStartScreenWidget::PVStartScreenWidget(PVMainWindow *mw, PVMainWindow *parent) : QWidget(parent)
{
	PVLOG_DEBUG("PVInspector::PVStartScreenWidget::%s\n", __FUNCTION__);

	main_window = mw;

	// SIZE STUFF
	setMinimumSize(500,600);
	QSizePolicy temp_size_policy = QSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::Expanding);
	
	// OBJECTNAME STUFF
	setObjectName("PVStartScreenWidget");
	
	
	// LAYOUT STUFF
	// We need a Layout for that Widget
	QHBoxLayout *main_layout = new QHBoxLayout(this);
	// We fix the margins for that Layout
	main_layout->setContentsMargins(0,0,0,0);
	
	// We also need two Layout for the left and right parts
	QVBoxLayout *left_layout = new QVBoxLayout(this);
	left_layout->setContentsMargins(0,0,0,0);
	QVBoxLayout *right_layout = new QVBoxLayout(this);
	right_layout->setContentsMargins(0,0,0,0);
	
	// We add these two layouts to the main_layout
	main_layout->addLayout(left_layout,1);
	main_layout->addLayout(right_layout,1);
	
	// We create the three Widgets
	format_widget = new QWidget(this);
	format_widget->setObjectName("format_widget_of_PVStartScreenWidget");
	
	import_widget = new QWidget(this);
	//import_widget->setObjectName("import_widget_of_PVStartScreenWidget");
	
	investigation_widget = new QWidget(this);
	investigation_widget->setObjectName("investigation_widget_of_PVStartScreenWidget");
	
	// We put these Widgets in the right layouts and in the right order
	left_layout->addWidget(investigation_widget);
	right_layout->addWidget(import_widget);
	right_layout->addWidget(format_widget);


	// Each of these three Widget needs a Layout
	QVBoxLayout *format_widget_layout = new QVBoxLayout(format_widget);
	QVBoxLayout *import_widget_layout = new QVBoxLayout(import_widget);
	QVBoxLayout *investigation_widget_layout = new QVBoxLayout(investigation_widget);
	
	// We attach these Layouts to there respective Widgets
	format_widget->setLayout(format_widget_layout);
	import_widget->setLayout(import_widget_layout);
	investigation_widget->setLayout(investigation_widget_layout);
	
	// We create the headers labels
	QLabel *format_label = new QLabel("Format", format_widget);
	QLabel *import_label = new QLabel("Import", import_widget);
	QLabel *investigation_label = new QLabel("INVESTIGATION", investigation_widget);
	
	// We add the labels
	format_widget_layout->addWidget(format_label);
	import_widget_layout->addWidget(import_label);
	investigation_widget_layout->addWidget(investigation_label);
	
	// Buttons
	QPushButton *create_new_format_button = new QPushButton("Create a new format", format_widget);
	QPushButton *edit_format_button = new QPushButton("Edit a format", format_widget);
	QPushButton *import_file_button = new QPushButton("Import files", import_widget);
	QPushButton *import_from_database_button = new QPushButton("Import from a database...", import_widget);
	QPushButton *create_new_investigation_button = new QPushButton("Create a new investigation", investigation_widget);
	QPushButton *open_investigation_button = new QPushButton("Open an investigation", investigation_widget);
	// adding these buttons to there parent widgets...
	format_widget_layout->addWidget(create_new_format_button);
	format_widget_layout->addWidget(edit_format_button);
	import_widget_layout->addWidget(import_file_button);
	import_widget_layout->addWidget(import_from_database_button);
	investigation_widget_layout->addWidget(create_new_investigation_button);
	investigation_widget_layout->addWidget(open_investigation_button);
	
	
	// Final Stretch as Spacer ...
	format_widget_layout->addStretch(1);
	import_widget_layout->addStretch(1);
	investigation_widget_layout->addStretch(1);
	
	setLayout(main_layout);
}