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
#include <QListWidget>
#include <QScrollBar>
#include <QFontMetrics>

#include <PVStartScreenWidget.h>

#include <picviz/PVSource.h>
#include <pvkernel/rush/PVSourceDescription.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/widgets/PVUtils.h>

void PVInspector::PVAddRecentItemFuncObserver::update(const arguments_deep_copy_type& args) const
{
	_parent->refresh_recent_items(std::get<1>(args));
}

void PVInspector::PVAddSourceRecentItemFuncObserver::update(const arguments_deep_copy_type&) const
{
	_parent->refresh_recent_items(PVCore::PVRecentItemsManager::Category::SOURCES);
}

/******************************************************************************
 *
 * PVInspector::PVStartScreenWidget::PVStartScreenWidget
 *
 *****************************************************************************/
PVInspector::PVStartScreenWidget::PVStartScreenWidget(PVMainWindow* parent) :
	QWidget(parent),
	_mw(parent),
	_recent_items_add_obs(this),
	_recent_items_add_source_obs(this)
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
		QVBoxLayout* recent_used_formats_layout = new QVBoxLayout();
		_recent_layouts[PVCore::PVRecentItemsManager::Category::USED_FORMATS] = recent_used_formats_layout;
		format_widget_layout->addLayout(recent_used_formats_layout);

		// edited
		QFrame* format_edited_widget_line = new QFrame(format_widget);
		format_edited_widget_line->setFrameShape(QFrame::HLine);
		QLabel *format_text_edited_label = new QLabel("Recent edited formats:", format_widget);
		format_text_edited_label->setObjectName("PVStartScreenWidget_text");
		format_widget_layout->addWidget(format_edited_widget_line);
		format_widget_layout->addWidget(format_text_edited_label);
		QVBoxLayout* recent_edited_formats_layout = new QVBoxLayout();
		_recent_layouts[PVCore::PVRecentItemsManager::Category::EDITED_FORMATS] = recent_edited_formats_layout;
		format_widget_layout->addLayout(recent_edited_formats_layout);

		// supported
		QFrame* format_supported_widget_line = new QFrame(format_widget);
		format_supported_widget_line->setFrameShape(QFrame::HLine);
		QLabel *format_text_supported_label = new QLabel("Supported formats:", format_widget);
		format_text_supported_label->setObjectName("PVStartScreenWidget_text");
		format_widget_layout->addWidget(format_supported_widget_line);
		format_widget_layout->addWidget(format_text_supported_label);

		QVBoxLayout* supported_formats_layout = new QVBoxLayout();
		format_widget_layout->addLayout(supported_formats_layout);
		_recent_layouts[PVCore::PVRecentItemsManager::Category::SUPPORTED_FORMATS] = supported_formats_layout;


	// projects (text and line)
	QFrame* project_widget_line = new QFrame(import_widget);
	project_widget_line->setFrameShape(QFrame::HLine);
	project_widget_layout->addWidget(project_widget_line);
	QLabel *project_text_label = new QLabel("Recent projects:", project_widget);
	project_text_label->setObjectName("PVStartScreenWidget_text");
	project_widget_layout->addWidget(project_text_label);
	QVBoxLayout* recent_projects_layout = new QVBoxLayout();
	project_widget_layout->addLayout(recent_projects_layout);
	_recent_layouts[PVCore::PVRecentItemsManager::Category::PROJECTS] = recent_projects_layout;

	// Imports (text and line)
	QFrame* import_widget_line = new QFrame(project_widget);
	import_widget_line->setFrameShape(QFrame::HLine);
	QLabel *import_text_label = new QLabel("Recent sources:", import_widget);
	import_text_label->setObjectName("PVStartScreenWidget_text");
	import_widget_layout->addWidget(import_widget_line);
	import_widget_layout->addWidget(import_text_label);
	QVBoxLayout* recent_imports_layout = new QVBoxLayout();
	import_widget_layout->addLayout(recent_imports_layout);
	QVBoxLayout* recent_sources_layout = new QVBoxLayout();
	import_widget_layout->addLayout(recent_sources_layout);
	_recent_layouts[PVCore::PVRecentItemsManager::Category::SOURCES] = recent_sources_layout;

	// Final Stretch as Spacer ...
	format_widget_layout->addStretch(1);
	import_widget_layout->addStretch(1);
	project_widget_layout->addStretch(1);
	
	// Connections
	connect(create_new_project_button, SIGNAL(clicked(bool)), _mw, SLOT(project_new_Slot()));
	connect(open_project_button, SIGNAL(clicked(bool)), _mw, SLOT(project_load_Slot()));
	connect(import_file_button, SIGNAL(clicked(bool)), _mw, SLOT(import_type_default_Slot()));
	connect(create_new_format_button, SIGNAL(clicked(bool)), _mw, SLOT(new_format_Slot()));
	connect(edit_format_button, SIGNAL(clicked(bool)), _mw, SLOT(open_format_Slot()));

	PVHive::get().register_func_observer(PVCore::PVRecentItemsManager::get(), _recent_items_add_obs);
	PVHive::get().register_func_observer(PVCore::PVRecentItemsManager::get(), _recent_items_add_source_obs);

	refresh_all_recent_items();
}

void PVInspector::PVStartScreenWidget::refresh_all_recent_items()
{
	for (int category = (int) PVCore::PVRecentItemsManager::Category::FIRST ; category < (int) PVCore::PVRecentItemsManager::Category::LAST; category++) {
		refresh_recent_items(category);
	}
}

void PVInspector::PVStartScreenWidget::refresh_recent_items(int cat)
{
	// Qt doesn't like custom types, here's why we are using an int for this slot...
	PVCore::PVRecentItemsManager::Category category = (PVCore::PVRecentItemsManager::Category) cat;

	// Clear layout
	QLayoutItem* item = nullptr;
	QVBoxLayout* recent_item_layout = _recent_layouts[category];
	if (recent_item_layout == nullptr) return;
	while ((item = recent_item_layout->takeAt(0)) != nullptr) {
		delete item->widget();
		delete item;
	}

	// CSS Widget
	QWidget* widget = new QWidget();
	widget->setObjectName(PVCore::PVRecentItemsManager::get()->get_key(category) + "_of_PVStartScreenWidget");
	QVBoxLayout* widget_layout = new QVBoxLayout();
	widget->setLayout(widget_layout);
	_recent_layouts[category]->addWidget(widget);

	QListWidget* list = new QListWidget();
	list->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	list->verticalScrollBar()->setObjectName("verticalScrollBar_of_PVListingView");

	connect(list, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(dispatch_action(QListWidgetItem*)));

	for (QVariant var : PVCore::PVRecentItemsManager::get()->get_list(category)) {
		// item + data
		QListWidgetItem* item = new QListWidgetItem(list);
		item->setData(Qt::UserRole, var);
		item->setData(Qt::UserRole+1, cat);
		QString formated_string = get_string_from_variant(category, var);

		// Icon
		QFileInfo finfo(formated_string);
		QFileIconProvider ficon;
		QIcon icon = ficon.icon(finfo);
		item->setIcon(icon);

		// Text
		item->setText(PVWidgets::PVUtils::shorten_path(formated_string, list->font(), 275));
		item->setToolTip(formated_string);
	}

	widget_layout->addWidget(list);
}

QString PVInspector::PVStartScreenWidget::get_string_from_variant(PVCore::PVRecentItemsManager::Category category, const QVariant& var)
{
	switch (category)
	{
		case PVCore::PVRecentItemsManager::Category::PROJECTS:
		case PVCore::PVRecentItemsManager::Category::USED_FORMATS:
		case PVCore::PVRecentItemsManager::Category::EDITED_FORMATS:
		{
			return var.toString();
		}
		case PVCore::PVRecentItemsManager::Category::SOURCES:
		{
			return get_string_from_source_description(var);
		}
		case PVCore::PVRecentItemsManager::Category::SUPPORTED_FORMATS:
		{
			return get_string_from_format(var);
		}
		default:
		{
			break;
		}
	}
}

QString PVInspector::PVStartScreenWidget::get_string_from_format(const QVariant& var)
{
	PVRush::PVFormat format = var.value<PVRush::PVFormat>();

	return QString("%1 (%2)").arg(format.get_format_name()).arg(format.get_full_path());
}

QString PVInspector::PVStartScreenWidget::get_string_from_source_description(const QVariant& var)
{
	PVRush::PVSourceDescription src_desc = var.value<PVRush::PVSourceDescription>();

	QStringList inputs_string;
	for (auto input : src_desc.get_inputs()) {
		inputs_string << input->human_name();
	}
	return inputs_string.join(", ");
}

void PVInspector::PVStartScreenWidget::dispatch_action(QListWidgetItem* item)
{
	QVariant var = item->data(Qt::UserRole);
	PVCore::PVRecentItemsManager::Category category = (PVCore::PVRecentItemsManager::Category) item->data(Qt::UserRole+1).toInt();

	switch (category)
	{
		case PVCore::PVRecentItemsManager::Category::PROJECTS:
		{
			_mw->load_project(var.toString());
			break;
		}
		case PVCore::PVRecentItemsManager::Category::SOURCES:
		{
			PVRush::PVSourceDescription src_desc = var.value<PVRush::PVSourceDescription>();
			Picviz::PVSource_p src_p = Picviz::PVSource::create_source_from_description(_mw->_scene, src_desc);
			_mw->load_source(src_p);
			break;
		}
		case PVCore::PVRecentItemsManager::Category::EDITED_FORMATS:
		case PVCore::PVRecentItemsManager::Category::USED_FORMATS:
		{
			_mw->edit_format_Slot(var.toString());
			break;
		}
		case PVCore::PVRecentItemsManager::Category::SUPPORTED_FORMATS:
		{
			_mw->edit_format_Slot(var.value<PVRush::PVFormat>().get_full_path());
			break;
		}
		default:
		{
			break;
		}
	}
}
