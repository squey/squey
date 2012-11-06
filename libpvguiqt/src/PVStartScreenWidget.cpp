/**
 * \file PVStartScreenWidget.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */


#include <pvkernel/core/general.h>
#include <picviz/PVView.h>

#include <QApplication>
#include <QFileInfo>
#include <QFileIconProvider>
#include <QIcon>
#include <QScrollBar>
#include <QFontMetrics>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSpacerItem>

#include <pvguiqt/PVStartScreenWidget.h>

#include <picviz/PVSource.h>

#include <pvkernel/rush/PVSourceDescription.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/widgets/PVUtils.h>

#include <pvguiqt/PVInputTypeMenuEntries.h>

void PVGuiQt::PVAddRecentItemFuncObserver::update(const arguments_deep_copy_type& args) const
{
	_parent->refresh_recent_items(std::get<1>(args));
}

void PVGuiQt::PVAddSourceRecentItemFuncObserver::update(const arguments_deep_copy_type&) const
{
	_parent->refresh_recent_items(PVCore::PVRecentItemsManager::Category::SOURCES);
}

/******************************************************************************
 *
 * PVGuiQt::PVStartScreenWidget::PVStartScreenWidget
 *
 *****************************************************************************/
PVGuiQt::PVStartScreenWidget::PVStartScreenWidget(QWidget* parent) :
	QWidget(parent),
	_recent_items_add_obs(this),
	_recent_items_add_source_obs(this)
{
	PVLOG_DEBUG("PVGuiQt::PVStartScreenWidget::%s\n", __FUNCTION__);

	// SIZE STUFF
	setMinimumSize(500,600);
	
	QHBoxLayout *main_layout = new QHBoxLayout();

	QPixmap* pv_welcomeIcon = new QPixmap(":/start-logo");

	QLabel* pv_labelWelcomeIcon = new QLabel();
	pv_labelWelcomeIcon->setPixmap(*pv_welcomeIcon);
	pv_labelWelcomeIcon->resize(pv_welcomeIcon->width(), pv_welcomeIcon->height());

	QVBoxLayout* pv_startLayout = new QVBoxLayout(this);
	pv_startLayout->addItem(new QSpacerItem(1, 1, QSizePolicy::Minimum, QSizePolicy::Expanding));
	QVBoxLayout* centerLayout = new QVBoxLayout();
	centerLayout->setAlignment(Qt::AlignHCenter);
	centerLayout->addWidget(pv_labelWelcomeIcon);
	pv_startLayout->addLayout(centerLayout);
	pv_startLayout->addItem(new QSpacerItem(1, 1, QSizePolicy::Minimum, QSizePolicy::Expanding));

	QWidget* start_widget = new QWidget();
	start_widget->setObjectName("PVStartScreenWidget");
	start_widget->setLayout(main_layout);
	pv_startLayout->addWidget(start_widget);

	QGridLayout* versionLayout = new QGridLayout();
	QLabel* label = new QLabel(tr("Current version") + QString(" :"));
	label->setAlignment(Qt::AlignRight);
	versionLayout->addWidget(label, 0, 0);
	label = new QLabel(QString(PICVIZ_CURRENT_VERSION_STR));
	label->setAlignment(Qt::AlignRight);
	versionLayout->addWidget(label, 0, 2);
	label = new QLabel(tr("Last version of the %1.%2 branch").arg(PICVIZ_CURRENT_VERSION_MAJOR).arg(PICVIZ_CURRENT_VERSION_MINOR) + QString(" :"));
	label->setAlignment(Qt::AlignRight);
	versionLayout->addWidget(label, 2, 0);
	QLabel* pv_lastCurVersion = new QLabel("N/A");
	pv_lastCurVersion->setAlignment(Qt::AlignRight);
	versionLayout->addWidget(pv_lastCurVersion, 2, 2);
	label = new QLabel(tr("Last major version") + QString(" :"));
	label->setAlignment(Qt::AlignRight);
	versionLayout->addWidget(label, 4, 0);
	QLabel* pv_lastMajVersion = new QLabel("N/A");
	pv_lastMajVersion->setAlignment(Qt::AlignRight);
	versionLayout->addWidget(pv_lastMajVersion, 4, 2);

	QHBoxLayout* hboxVersionLayout = new QHBoxLayout();
	hboxVersionLayout->addItem(new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Minimum));
	hboxVersionLayout->addLayout(versionLayout);

	pv_startLayout->addLayout(hboxVersionLayout);

	//setLayout(pv_startLayout);

	// OBJECTNAME STUFF
	
	// LAYOUT STUFF
	// We need a Layout for that Widget

	// We fix the margins for that Layout
	main_layout->setContentsMargins(0,0,0,0);

	// We also need two Layout for the left and right parts
	QVBoxLayout *left_layout = new QVBoxLayout();
	left_layout->setContentsMargins(0,0,0,0);
	QVBoxLayout *right_layout = new QVBoxLayout();
	right_layout->setContentsMargins(0,0,0,0);
	
	// We add these two layouts to the main_layout
	main_layout->addLayout(left_layout, 1);
	main_layout->addLayout(right_layout, 1);


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

	QPushButton *create_new_project_button = new QPushButton("Create a new project");
	create_new_project_button->setIcon(QIcon(":/new-icon-white"));
	create_new_project_button->setCursor(Qt::PointingHandCursor);

	QPushButton *open_project_button = new QPushButton("Open a project");
	open_project_button->setIcon(QIcon(":/open-icon-white"));
	open_project_button->setCursor(Qt::PointingHandCursor);

	// adding these buttons to their parent widgets...
	format_widget_layout->addWidget(create_new_format_button);
	format_widget_layout->addWidget(edit_format_button);

	// Import buttons
	PVGuiQt::PVInputTypeMenuEntries::add_inputs_to_layout(import_widget_layout, this, SLOT(import_type_Slot()));

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
		format_widget_layout->addLayout(recent_used_formats_layout);
		custom_listwidget_t* recent_used_formats_list = new custom_listwidget_t();
		recent_used_formats_layout->addWidget(recent_used_formats_list);
		recent_used_formats_list->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
		recent_used_formats_list->verticalScrollBar()->setObjectName("verticalScrollBar_of_PVListingView");
		_recent_list_widgets[PVCore::PVRecentItemsManager::Category::USED_FORMATS] = recent_used_formats_list;

		// edited
		QFrame* format_edited_widget_line = new QFrame(format_widget);
		format_edited_widget_line->setFrameShape(QFrame::HLine);
		QLabel *format_text_edited_label = new QLabel("Recent edited formats:", format_widget);
		format_text_edited_label->setObjectName("PVStartScreenWidget_text");
		format_widget_layout->addWidget(format_edited_widget_line);
		format_widget_layout->addWidget(format_text_edited_label);
		QVBoxLayout* recent_edited_formats_layout = new QVBoxLayout();
		format_widget_layout->addLayout(recent_edited_formats_layout);
		custom_listwidget_t* recent_edited_formats_list = new custom_listwidget_t();
		recent_edited_formats_layout->addWidget(recent_edited_formats_list);
		recent_edited_formats_list->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
		recent_edited_formats_list->verticalScrollBar()->setObjectName("verticalScrollBar_of_PVListingView");
		_recent_list_widgets[PVCore::PVRecentItemsManager::Category::EDITED_FORMATS] = recent_edited_formats_list;

		// supported
		QFrame* format_supported_widget_line = new QFrame(format_widget);
		format_supported_widget_line->setFrameShape(QFrame::HLine);
		QLabel *format_text_supported_label = new QLabel("Supported formats:", format_widget);
		format_text_supported_label->setObjectName("PVStartScreenWidget_text");
		format_widget_layout->addWidget(format_supported_widget_line);
		format_widget_layout->addWidget(format_text_supported_label);

		QVBoxLayout* supported_formats_layout = new QVBoxLayout();
		format_widget_layout->addLayout(supported_formats_layout);
		custom_listwidget_t* supported_formats_list = new custom_listwidget_t();
		supported_formats_layout->addWidget(supported_formats_list);
		supported_formats_list->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
		supported_formats_list->verticalScrollBar()->setObjectName("verticalScrollBar_of_PVListingView");
		_recent_list_widgets[PVCore::PVRecentItemsManager::Category::SUPPORTED_FORMATS] = supported_formats_list;


	// projects (text and line)
	QFrame* project_widget_line = new QFrame(import_widget);
	project_widget_line->setFrameShape(QFrame::HLine);
	project_widget_layout->addWidget(project_widget_line);
	QLabel *project_text_label = new QLabel("Recent projects:", project_widget);
	project_text_label->setObjectName("PVStartScreenWidget_text");
	project_widget_layout->addWidget(project_text_label);
	QVBoxLayout* recent_projects_layout = new QVBoxLayout();
	project_widget_layout->addLayout(recent_projects_layout);
	custom_listwidget_t* recent_projects_list = new custom_listwidget_t();
	recent_projects_layout->addWidget(recent_projects_list);
	recent_projects_list->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	recent_projects_list->verticalScrollBar()->setObjectName("verticalScrollBar_of_PVListingView");
	_recent_list_widgets[PVCore::PVRecentItemsManager::Category::PROJECTS] = recent_projects_list;

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
	custom_listwidget_t* import_list = new custom_listwidget_t();
	recent_sources_layout->addWidget(import_list);
	import_list->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	import_list->verticalScrollBar()->setObjectName("verticalScrollBar_of_PVListingView");
	_recent_list_widgets[PVCore::PVRecentItemsManager::Category::SOURCES] = import_list;

	_item_font = import_list->font();

	// Final Stretch as Spacer ...
	format_widget_layout->addStretch(1);
	import_widget_layout->addStretch(1);
	project_widget_layout->addStretch(1);
	
	// Connections
	connect(create_new_project_button, SIGNAL(clicked(bool)), this, SIGNAL(new_project()));
	connect(open_project_button, SIGNAL(clicked(bool)), this, SIGNAL(load_project()));
	connect(create_new_format_button, SIGNAL(clicked(bool)), this, SIGNAL(new_format()));
	connect(edit_format_button, SIGNAL(clicked(bool)), this, SIGNAL(load_format()));

	PVHive::get().register_func_observer(PVCore::PVRecentItemsManager::get(), _recent_items_add_obs);
	PVHive::get().register_func_observer(PVCore::PVRecentItemsManager::get(), _recent_items_add_source_obs);

	refresh_all_recent_items();
}

void PVGuiQt::PVStartScreenWidget::import_type_Slot()
{
	QAction* action_src = (QAction*) sender();
	QString const& itype = action_src->data().toString();
	emit import_type(itype);
}

void PVGuiQt::PVStartScreenWidget::refresh_all_recent_items()
{
	for (int category = (int) PVCore::PVRecentItemsManager::Category::FIRST ; category < (int) PVCore::PVRecentItemsManager::Category::LAST; category++) {
		refresh_recent_items(category);
	}
}

void PVGuiQt::PVStartScreenWidget::refresh_recent_items(int cat)
{
	// Qt doesn't like custom types, here's why we are using an int for this slot...
	PVCore::PVRecentItemsManager::Category category = (PVCore::PVRecentItemsManager::Category) cat;

	custom_listwidget_t* list = _recent_list_widgets[category];
	list->setObjectName("RecentProjectItem");
	//list->setSizePolicy(QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Maximum));
	list->clear();

	uint64_t index = 0 ;
	for (QVariant var : PVCore::PVRecentItemsManager::get()->get_list(category)) {
		// item + data
		QListWidgetItem* item = new QListWidgetItem(list);
		item->setData(Qt::UserRole, var);
		item->setData(Qt::UserRole+1, cat);
		QString short_string;
		QString long_string;
		QStringList filenames;
		std::tie(short_string, long_string, filenames) = get_string_from_variant(category, var);

		QWidget* widget = new QWidget();
		QHBoxLayout* layout = new QHBoxLayout();
		layout->setAlignment(Qt::AlignLeft);
		widget->setLayout(layout);

		// Icon
		QLabel* icon_label = new QLabel();
		QIcon icon;
		if (filenames.size() == 1) {
			QFileInfo finfo(filenames[0]);
			QFileIconProvider ficon;
			icon = ficon.icon(finfo);
		}
		else {
			icon = QApplication::style()->standardIcon(QStyle::SP_FileDialogNewFolder);
		}
		icon_label->setPixmap(icon.pixmap(15, 15));
		layout->addWidget(icon_label);

		// Text
		QLabel* text_label = new QLabel();
		text_label->setTextFormat(Qt::RichText);
		text_label->setText(QString("<a href=\"%1;%2\">" + short_string + "</a>").arg(cat).arg(index));
		text_label->setToolTip(long_string);
		connect(text_label, SIGNAL(linkActivated(const QString &)), this, SLOT(dispatch_action(const QString &)));
		layout->addWidget(text_label);

		item->setSizeHint(QSize(widget->sizeHint().width(), widget->sizeHint().height()-6)); // Do not forget this!
		list->setItemWidget(item, widget);

		index++;
	}
}

PVGuiQt::PVStartScreenWidget::descr_strings_t PVGuiQt::PVStartScreenWidget::get_string_from_variant(
	PVCore::PVRecentItemsManager::Category category,
	const QVariant& var
)
{
	switch (category)
	{
		case PVCore::PVRecentItemsManager::Category::PROJECTS:
		case PVCore::PVRecentItemsManager::Category::USED_FORMATS:
		case PVCore::PVRecentItemsManager::Category::EDITED_FORMATS:
		{
			QString long_string = var.toString();
			QString short_string = PVWidgets::PVUtils::shorten_path(long_string, _item_font, _item_width);
			QStringList filenames;
			filenames << long_string;
			return std::make_tuple(short_string, long_string, filenames);
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
			assert(false);
			return descr_strings_t();
		}
	}
}

PVGuiQt::PVStartScreenWidget::descr_strings_t PVGuiQt::PVStartScreenWidget::get_string_from_format(const QVariant& var)
{
	PVRush::PVFormat format = var.value<PVRush::PVFormat>();

	QString long_string = QString("%1 (%2)").arg(format.get_format_name()).arg(format.get_full_path());
	QString short_string = PVWidgets::PVUtils::shorten_path(long_string, _item_font, _item_width);
	QStringList filenames;
	filenames << format.get_full_path();

	return std::make_tuple(short_string, long_string, filenames);
}

PVGuiQt::PVStartScreenWidget::descr_strings_t PVGuiQt::PVStartScreenWidget::get_string_from_source_description(const QVariant& var)
{
	PVRush::PVSourceDescription src_desc = var.value<PVRush::PVSourceDescription>();

	QString long_string;
	QString short_string;
	QStringList filenames;

	QStringList short_strings;
	if (src_desc.get_inputs().size() == 1) {
		QString source_path = src_desc.get_inputs()[0]->human_name();
		short_string = PVWidgets::PVUtils::shorten_path(source_path, _item_font, _item_width) + " [" + src_desc.get_format().get_format_name() +"]";
		long_string = source_path + " [" + src_desc.get_format().get_format_name() +"]";
		filenames << source_path;
	}
	else {
		for (auto input : src_desc.get_inputs()) {
			QFileInfo info(input->human_name());
			short_strings << info.fileName();
			filenames << input->human_name();
		}
		short_string = short_strings.join(", ") + " [" + src_desc.get_format().get_format_name() +"]";
		long_string = "[" + src_desc.get_format().get_format_name() +"]\n" + filenames.join("\n");
	}

	return std::make_tuple(short_string, long_string, filenames);
}

void PVGuiQt::PVStartScreenWidget::dispatch_action(const QString& id)
{
	// This is kind of a hack but it saves the use of a QAbstractListModel/QListView...
	QStringList ids = id.split(";");
	PVCore::PVRecentItemsManager::Category category = (PVCore::PVRecentItemsManager::Category) ids[0].toUInt();
	uint64_t item_index = ids[1].toUInt();
	QListWidgetItem* item = _recent_list_widgets[category]->item(item_index);

	QVariant var = item->data(Qt::UserRole);

	switch (category)
	{
		case PVCore::PVRecentItemsManager::Category::PROJECTS:
		{
			emit load_project_from_path(var.toString());
			break;
		}
		case PVCore::PVRecentItemsManager::Category::SOURCES:
		{
			PVRush::PVSourceDescription src_desc = var.value<PVRush::PVSourceDescription>();
			emit load_source_from_description(src_desc);
			break;
		}
		case PVCore::PVRecentItemsManager::Category::EDITED_FORMATS:
		case PVCore::PVRecentItemsManager::Category::USED_FORMATS:
		{
			emit edit_format(var.toString());
			break;
		}
		case PVCore::PVRecentItemsManager::Category::SUPPORTED_FORMATS:
		{
			emit edit_format(var.toString());
			break;
		}
		default:
		{
			break;
		}
	}
}
