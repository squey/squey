/**
 * \file PVStartScreenWidget.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */


#include <pvkernel/core/general.h>
#include <picviz/PVView.h>

#include <PVMainWindow.h>
#include <PVTabSplitter.h>

#include <QFileInfo>
#include <QFileIconProvider>
#include <QIcon>

#include <PVStartScreenWidget.h>

/******************************************************************************
 *
 * PVInspector::PVStartScreenWidget::PVStartScreenWidget
 *
 *****************************************************************************/
PVInspector::PVStartScreenWidget::PVStartScreenWidget(PVMainWindow* parent) : QWidget(parent), _mw(parent)
{
	PVLOG_DEBUG("PVInspector::PVStartScreenWidget::%s\n", __FUNCTION__);

	// SIZE STUFF
	setMinimumSize(500,600);
	QSizePolicy temp_size_policy = QSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::Expanding);
	
	// OBJECTNAME STUFF
	
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
	main_layout->addLayout(left_layout, 1);
	main_layout->addLayout(right_layout, 1);

	setObjectName("PVStartScreenWidget");

	// We create the three Widgets
	format_widget = new QWidget(this);
	format_widget->setObjectName("format_widget_of_PVStartScreenWidget");
	
	import_widget = new QWidget(this);
	import_widget->setObjectName("import_widget_of_PVStartScreenWidget");
	
	project_widget = new QWidget(this);
	project_widget->setObjectName("project_widget_of_PVStartScreenWidget");
	
	// We put these Widgets in the right layouts and in the right order
	left_layout->addWidget(project_widget);
	left_layout->addWidget(import_widget);
	right_layout->addWidget(format_widget);


	// Each of these three Widget needs a Layout
	QVBoxLayout *format_widget_layout = new QVBoxLayout();
	QVBoxLayout *import_widget_layout = new QVBoxLayout();
	QVBoxLayout *project_widget_layout = new QVBoxLayout();
	
	// We attach these Layouts to there respective Widgets
	format_widget->setLayout(format_widget_layout);
	import_widget->setLayout(import_widget_layout);
	project_widget->setLayout(project_widget_layout);
	
	// We create the headers labels
	QLabel *format_label = new QLabel("FORMATS");
	format_label->setObjectName("PVStartScreenWidget_header");
	QLabel *import_label = new QLabel("IMPORT SOURCES");
	import_label->setObjectName("PVStartScreenWidget_header");
	QLabel *project_label = new QLabel("PROJECTS");
	project_label->setObjectName("PVStartScreenWidget_header");
	
	// We add the labels
	format_widget_layout->addWidget(format_label);
	import_widget_layout->addWidget(import_label);
	project_widget_layout->addWidget(project_label);
	
	// Buttons
	QPushButton *create_new_format_button = new QPushButton("Create a new format");
	create_new_format_button->setIcon(QIcon(":/new-icon-white"));
	create_new_format_button->setCursor(Qt::PointingHandCursor);

	QPushButton *edit_format_button = new QPushButton("Edit a format");
	edit_format_button->setIcon(QIcon(":/edit-icon-white"));
	edit_format_button->setCursor(Qt::PointingHandCursor);

	QPushButton *import_file_button = new QPushButton("Import files");
	import_file_button->setIcon(QIcon(":/new-icon-white"));
	import_file_button->setCursor(Qt::PointingHandCursor);

	QPushButton *import_from_database_button = new QPushButton("Import from database");
	import_from_database_button->setIcon(QIcon(":/open-icon-white"));
	import_from_database_button->setCursor(Qt::PointingHandCursor);

	QPushButton *create_new_project_button = new QPushButton("Create a new project");
	create_new_project_button->setIcon(QIcon(":/new-icon-white"));
	create_new_project_button->setCursor(Qt::PointingHandCursor);

	QPushButton *open_project_button = new QPushButton("Open a project");
	open_project_button->setIcon(QIcon(":/open-icon-white"));
	open_project_button->setCursor(Qt::PointingHandCursor);

	// adding these buttons to their parent widgets...
	format_widget_layout->addWidget(create_new_format_button);
	format_widget_layout->addWidget(edit_format_button);
	import_widget_layout->addWidget(import_file_button);
	import_widget_layout->addWidget(import_from_database_button);
	project_widget_layout->addWidget(create_new_project_button);
	project_widget_layout->addWidget(open_project_button);


	// Formats (texts and lines)
		// used
		QFrame* format_used_widget_line = new QFrame(format_widget);
		format_used_widget_line->setFrameShape(QFrame::HLine);
		QLabel *format_text_used_label = new QLabel("Recent used formats:", format_widget);
		format_text_used_label->setObjectName("PVStartScreenWidget_text");
		format_widget_layout->addWidget(format_used_widget_line);
		format_widget_layout->addWidget(format_text_used_label);

		// edited
		QFrame* format_edited_widget_line = new QFrame(format_widget);
		format_edited_widget_line->setFrameShape(QFrame::HLine);
		QLabel *format_text_edited_label = new QLabel("Recent edited formats:", format_widget);
		format_text_edited_label->setObjectName("PVStartScreenWidget_text");
		format_widget_layout->addWidget(format_edited_widget_line);
		format_widget_layout->addWidget(format_text_edited_label);

		// detected
		QFrame* format_detected_widget_line = new QFrame(format_widget);
		format_detected_widget_line->setFrameShape(QFrame::HLine);
		QLabel *format_text_detected_label = new QLabel("Recent detected formats:", format_widget);
		format_text_detected_label->setObjectName("PVStartScreenWidget_text");
		format_widget_layout->addWidget(format_detected_widget_line);
		format_widget_layout->addWidget(format_text_detected_label);

	// projects (text and line)
	QFrame* project_widget_line = new QFrame(import_widget);
	project_widget_line->setFrameShape(QFrame::HLine);
	project_widget_layout->addWidget(project_widget_line);
	QLabel *project_text_label = new QLabel("Recent projects:", project_widget);
	project_text_label->setObjectName("PVStartScreenWidget_text");
	project_widget_layout->addWidget(project_text_label);
	_recent_projects_layout = new QVBoxLayout();
	project_widget_layout->addLayout(_recent_projects_layout);

	// Imports (text and line)
	QFrame* import_widget_line = new QFrame(project_widget);
	import_widget_line->setFrameShape(QFrame::HLine);
	QLabel *import_text_label = new QLabel("Recent sources:", import_widget);
	import_text_label->setObjectName("PVStartScreenWidget_text");
	import_widget_layout->addWidget(import_widget_line);
	import_widget_layout->addWidget(import_text_label);
	
	// Final Stretch as Spacer ...
	format_widget_layout->addStretch(1);
	import_widget_layout->addStretch(1);
	project_widget_layout->addStretch(1);
	
	// Connections
	connect(create_new_project_button, SIGNAL(clicked(bool)), _mw, SLOT(project_new_Slot()));
	connect(open_project_button, SIGNAL(clicked(bool)), _mw, SLOT(project_load_Slot()));
	connect(import_file_button, SIGNAL(clicked(bool)), _mw, SLOT(import_type_default_Slot()));
	//connect(import_from_database_button, SIGNAL(clicked(bool)), _mw, SLOT(project_load_Slot()));
	connect(create_new_format_button, SIGNAL(clicked(bool)), _mw, SLOT(new_format_Slot()));

	update_recent_projects();
}

void PVInspector::PVStartScreenWidget::update_recent_projects()
{
	// Clear layout
	QLayoutItem* item = nullptr;
	while ((item = _recent_projects_layout->takeAt(0)) != nullptr) {
		delete item->widget();
		delete item;
	}

	// CSS Widget
	QWidget* widget = new QWidget();
	widget->setObjectName("RecentProjectItem");
	QVBoxLayout* widget_layout = new QVBoxLayout();
	widget->setLayout(widget_layout);
	_recent_projects_layout->addWidget(widget);

	// Add recent items
	for (QString project_filename : _mw->get_recent_projects_list()) {

		// Icon
		QFileInfo finfo(project_filename);
		QFileIconProvider ficon;
		QIcon icon = ficon.icon(finfo);
		QLabel* project_icon = new QLabel();
		project_icon->setPixmap(icon.pixmap(20, 20));

		// Label
		QLabel* project_label = new QLabel();
		project_label->setTextFormat(Qt::RichText);
		project_label->setText(QString("<a href=\"%1\">%2</a>").arg(project_filename).arg(project_filename));
		connect(project_label, SIGNAL(linkActivated(const QString &)), _mw, SLOT(load_project(const QString &)));

		// Layout
		QHBoxLayout* layout = new QHBoxLayout();
		layout->addWidget(project_icon);
		layout->addWidget(project_label);
		layout->addStretch();
		widget_layout->addLayout(layout);
	}
}
