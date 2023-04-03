//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <cassert>

#include <QApplication>
#include <QCheckBox>
#include <QFileInfo>
#include <QFileIconProvider>
#include <QIcon>
#include <QScrollBar>
#include <QFontMetrics>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QSpacerItem>
#include <QDialogButtonBox>

#include <pvguiqt/PVStartScreenWidget.h>
#include <pvguiqt/PVInputTypeMenuEntries.h>
#include <pvguiqt/PVCredentialDialog.h>

#include <pvkernel/rush/PVSourceDescription.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/widgets/PVUtils.h>

#include INENDI_VERSION_FILE_PATH

/******************************************************************************
 *
 * PVGuiQt::PVStartScreenWidget::PVStartScreenWidget
 *
 *****************************************************************************/
const QFont* PVGuiQt::PVStartScreenWidget::_item_font = nullptr;

namespace
{
struct connecter {
	template <PVCore::Category c>
	void call() const
	{
		PVCore::PVRecentItemsManager::get()._add_item[c].connect(
		    sigc::mem_fun(&sc, &PVGuiQt::PVStartScreenWidget::refresh_recent_items<c>));
	}

	PVGuiQt::PVStartScreenWidget& sc;
};
} // namespace

PVGuiQt::PVStartScreenWidget::PVStartScreenWidget(QWidget* parent) : QWidget(parent)
{
	PVLOG_DEBUG("PVGuiQt::PVStartScreenWidget::%s\n", __FUNCTION__);

	// SIZE STUFF
	setMinimumSize(500, 600);

	auto main_layout = new QHBoxLayout();

	QPixmap pv_welcomeIcon(":/start-logo");

	auto pv_labelWelcomeIcon = new QLabel();
	pv_labelWelcomeIcon->setPixmap(pv_welcomeIcon);
	pv_labelWelcomeIcon->resize(pv_welcomeIcon.width(), pv_welcomeIcon.height());

	auto pv_startLayout = new QVBoxLayout(this);
	pv_startLayout->addItem(new QSpacerItem(1, 1, QSizePolicy::Minimum, QSizePolicy::Expanding));
	auto centerLayout = new QVBoxLayout();
	centerLayout->setAlignment(Qt::AlignHCenter);
	centerLayout->addWidget(pv_labelWelcomeIcon);
	pv_startLayout->addLayout(centerLayout);
	pv_startLayout->addItem(new QSpacerItem(1, 1, QSizePolicy::Minimum, QSizePolicy::Expanding));

	auto start_widget = new QWidget();
	start_widget->setObjectName("PVStartScreenWidget");
	start_widget->setLayout(main_layout);
	pv_startLayout->addWidget(start_widget);

	auto versionLayout = new QGridLayout();
	QLabel* label = new QLabel(tr("Current version") + QString(" :"));
	label->setAlignment(Qt::AlignRight);
	versionLayout->addWidget(label, 0, 0);
	label = new QLabel(QString(INENDI_CURRENT_VERSION_STR));
	label->setAlignment(Qt::AlignRight);
	versionLayout->addWidget(label, 0, 2);

	auto hboxVersionLayout = new QHBoxLayout();
	hboxVersionLayout->addItem(new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Minimum));
	hboxVersionLayout->addLayout(versionLayout);

	pv_startLayout->addLayout(hboxVersionLayout);

	// setLayout(pv_startLayout);

	// OBJECTNAME STUFF

	// LAYOUT STUFF
	// We need a Layout for that Widget

	// We fix the margins for that Layout
	main_layout->setContentsMargins(0, 0, 0, 0);

	// We also need two Layout for the left and right parts
	auto left_layout = new QVBoxLayout();
	left_layout->setContentsMargins(0, 0, 0, 0);
	auto right_layout = new QVBoxLayout();
	right_layout->setContentsMargins(0, 0, 0, 0);

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
	auto format_widget_layout = new QVBoxLayout();
	auto import_widget_layout = new QVBoxLayout();
	auto project_widget_layout = new QVBoxLayout();

	// We attach these Layouts to there respective Widgets
	format_widget->setLayout(format_widget_layout);
	import_widget->setLayout(import_widget_layout);
	project_widget->setLayout(project_widget_layout);

	// We create the headers labels
	QLabel* format_label = new QLabel("FORMATS");
	format_label->setObjectName("PVStartScreenWidget_header");
	QLabel* import_label = new QLabel("SOURCES");
	import_label->setObjectName("PVStartScreenWidget_header");
	QLabel* project_label = new QLabel("INVESTIGATIONS");
	project_label->setObjectName("PVStartScreenWidget_header");

	// We add the labels
	format_widget_layout->addWidget(format_label);
	import_widget_layout->addWidget(import_label);
	project_widget_layout->addWidget(project_label);

	// Buttons
	QPushButton* create_new_format_button = new QPushButton("Create a new format...");
	create_new_format_button->setIcon(QIcon(":/new-icon-white"));
	create_new_format_button->setCursor(Qt::PointingHandCursor);

	QPushButton* edit_format_button = new QPushButton("Edit a format...");
	edit_format_button->setIcon(QIcon(":/edit-icon-white"));
	edit_format_button->setCursor(Qt::PointingHandCursor);

	QPushButton* create_new_project_button = new QPushButton("Create a new investigation");
	create_new_project_button->setIcon(QIcon(":/new-icon-white"));
	create_new_project_button->setCursor(Qt::PointingHandCursor);

	QPushButton* open_project_button = new QPushButton("Open an investigation...");
	open_project_button->setIcon(QIcon(":/open-icon-white"));
	open_project_button->setCursor(Qt::PointingHandCursor);

	// adding these buttons to their parent widgets...
	format_widget_layout->addWidget(create_new_format_button);
	format_widget_layout->addWidget(edit_format_button);

	// Import buttons
	auto hl = new QHBoxLayout();
	import_widget_layout->addLayout(hl);
	PVGuiQt::PVInputTypeMenuEntries::add_inputs_to_layout(hl, this, SLOT(import_type()));

	project_widget_layout->addWidget(create_new_project_button);
	project_widget_layout->addWidget(open_project_button);

	// Formats (texts and lines)
	// used
	auto format_used_widget_line = new QFrame(format_widget);
	format_used_widget_line->setFrameShape(QFrame::HLine);
	QLabel* format_text_used_label = new QLabel("Recent used formats:", format_widget);
	format_text_used_label->setObjectName("PVStartScreenWidget_text");
	format_widget_layout->addWidget(format_used_widget_line);
	format_widget_layout->addWidget(format_text_used_label);
	auto used_format_header_layout = new QHBoxLayout();
	QPushButton* clear_used_format_history = new QPushButton("Clear");
	clear_used_format_history->setObjectName("PVStartScreenWidget_clearHistoryButton");
	clear_used_format_history->setFocusPolicy(Qt::NoFocus);
	clear_used_format_history->setCursor(Qt::PointingHandCursor);
	connect(clear_used_format_history, &QPushButton::clicked,
	        [&] { clear_history_dlg<PVCore::Category::USED_FORMATS>(); });
	used_format_header_layout->addWidget(format_text_used_label);
	used_format_header_layout->addStretch();
	used_format_header_layout->addWidget(clear_used_format_history);
	auto recent_used_formats_layout = new QVBoxLayout();
	format_widget_layout->addLayout(used_format_header_layout);
	format_widget_layout->addLayout(recent_used_formats_layout);
	auto recent_used_formats_list = new custom_listwidget_t();
	recent_used_formats_layout->addWidget(recent_used_formats_list);
	recent_used_formats_list->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	recent_used_formats_list->verticalScrollBar()->setObjectName(
	    "verticalScrollBar_of_PVListingView");
	_recent_list_widgets[PVCore::Category::USED_FORMATS] = recent_used_formats_list;
	_recent_push_buttons[PVCore::Category::USED_FORMATS] = clear_used_format_history;

	// edited
	auto format_edited_widget_line = new QFrame(format_widget);
	format_edited_widget_line->setFrameShape(QFrame::HLine);
	QLabel* format_text_edited_label = new QLabel("Recent edited formats:", format_widget);
	format_text_edited_label->setObjectName("PVStartScreenWidget_text");
	format_widget_layout->addWidget(format_edited_widget_line);
	auto edited_format_header_layout = new QHBoxLayout();
	QPushButton* clear_edited_format_history = new QPushButton("Clear");
	clear_edited_format_history->setObjectName("PVStartScreenWidget_clearHistoryButton");
	clear_edited_format_history->setFocusPolicy(Qt::NoFocus);
	clear_edited_format_history->setCursor(Qt::PointingHandCursor);
	connect(clear_edited_format_history, &QPushButton::clicked,
	        [&] { clear_history_dlg<PVCore::Category::EDITED_FORMATS>(); });
	edited_format_header_layout->addWidget(format_text_edited_label);
	edited_format_header_layout->addStretch();
	edited_format_header_layout->addWidget(clear_edited_format_history);
	format_widget_layout->addLayout(edited_format_header_layout);
	auto recent_edited_formats_layout = new QVBoxLayout();
	format_widget_layout->addLayout(recent_edited_formats_layout);
	auto recent_edited_formats_list = new custom_listwidget_t();
	recent_edited_formats_layout->addWidget(recent_edited_formats_list);
	recent_edited_formats_list->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	recent_edited_formats_list->verticalScrollBar()->setObjectName(
	    "verticalScrollBar_of_PVListingView");
	_recent_list_widgets[PVCore::Category::EDITED_FORMATS] = recent_edited_formats_list;
	_recent_push_buttons[PVCore::Category::EDITED_FORMATS] = clear_edited_format_history;

	// projects (text and line)
	auto project_widget_line = new QFrame(import_widget);
	project_widget_line->setFrameShape(QFrame::HLine);
	project_widget_layout->addWidget(project_widget_line);
	QLabel* project_text_label = new QLabel("Recent investigations:", project_widget);
	project_text_label->setObjectName("PVStartScreenWidget_text");
	auto projects_header_layout = new QHBoxLayout();
	QPushButton* clear_project_history = new QPushButton("Delete");
	clear_project_history->setObjectName("PVStartScreenWidget_clearHistoryButton");
	clear_project_history->setFocusPolicy(Qt::NoFocus);
	clear_project_history->setCursor(Qt::PointingHandCursor);
	connect(clear_project_history, &QPushButton::clicked, [&] { delete_investigation_dlg(); });
	project_widget_layout->addLayout(projects_header_layout);
	projects_header_layout->addWidget(project_text_label);
	projects_header_layout->addStretch();
	projects_header_layout->addWidget(clear_project_history);
	auto recent_projects_list = new custom_listwidget_t();
	project_widget_layout->addWidget(recent_projects_list);
	recent_projects_list->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	recent_projects_list->verticalScrollBar()->setObjectName("verticalScrollBar_of_PVListingView");
	_recent_list_widgets[PVCore::Category::PROJECTS] = recent_projects_list;
	_recent_push_buttons[PVCore::Category::PROJECTS] = clear_project_history;

	// Imports (text and line)
	auto import_widget_line = new QFrame(project_widget);
	import_widget_line->setFrameShape(QFrame::HLine);
	QLabel* import_text_label = new QLabel("Recent sources:", import_widget);
	auto sources_header_layout = new QHBoxLayout();
	import_text_label->setCursor(Qt::PointingHandCursor);
	QPushButton* clear_source_history = new QPushButton("Clear");
	clear_source_history->setFocusPolicy(Qt::NoFocus);
	clear_source_history->setObjectName("PVStartScreenWidget_clearHistoryButton");
	clear_source_history->setCursor(Qt::PointingHandCursor);
	connect(clear_source_history, &QPushButton::clicked,
	        [&] { clear_history_dlg<PVCore::Category::SOURCES>(); });
	import_text_label->setObjectName("PVStartScreenWidget_text");
	import_widget_layout->addWidget(import_widget_line);
	sources_header_layout->addWidget(import_text_label);
	sources_header_layout->addStretch();
	sources_header_layout->addWidget(clear_source_history);
	import_widget_layout->addLayout(sources_header_layout);
	auto recent_imports_layout = new QVBoxLayout();
	import_widget_layout->addLayout(recent_imports_layout);
	auto recent_sources_layout = new QVBoxLayout();
	import_widget_layout->addLayout(recent_sources_layout);
	auto import_list = new custom_listwidget_t();
	recent_sources_layout->addWidget(import_list);
	import_list->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	import_list->verticalScrollBar()->setObjectName("verticalScrollBar_of_PVListingView");
	_recent_list_widgets[PVCore::Category::SOURCES] = import_list;
	_recent_push_buttons[PVCore::Category::SOURCES] = clear_source_history;

	_item_font = &import_list->font();

	// Final Stretch as Spacer ...
	format_widget_layout->addStretch(1);
	import_widget_layout->addStretch(1);
	project_widget_layout->addStretch(1);

	// Connections
	connect(create_new_project_button, &QAbstractButton::clicked, this,
	        &PVStartScreenWidget::new_project);
	connect(open_project_button, &QAbstractButton::clicked, this,
	        &PVStartScreenWidget::load_project);
	connect(create_new_format_button, &QAbstractButton::clicked, this,
	        &PVStartScreenWidget::new_format);
	connect(edit_format_button, &QAbstractButton::clicked, this, &PVStartScreenWidget::load_format);

	PVCore::PVRecentItemsManager::apply_on_category(connecter{*this});

	refresh_all_recent_items();
}

void PVGuiQt::PVStartScreenWidget::import_type()
{
	QAction* action_src = (QAction*)sender();
	assert(action_src);
	QString const& itype = action_src->data().toString();
	Q_EMIT import_type(itype);
}

namespace
{
struct refresher {
	template <PVCore::Category c>
	void call() const
	{
		sc.refresh_recent_items<c>();
	}

	PVGuiQt::PVStartScreenWidget& sc;
};
} // namespace

void PVGuiQt::PVStartScreenWidget::refresh_all_recent_items()
{
	PVCore::PVRecentItemsManager::apply_on_category(refresher{*this});
}

void PVGuiQt::PVStartScreenWidget::dispatch_action(const QString& id)
{
	// This is kind of a hack but it saves the use of a
	// QAbstractListModel/QListView...
	QStringList ids = id.split(";");
	PVCore::Category category = (PVCore::Category)ids[0].toUInt();
	uint64_t item_index = ids[1].toUInt();
	QListWidgetItem* item = _recent_list_widgets[category]->item(item_index);

	QVariant var = item->data(Qt::UserRole);

	switch (category) {
	case PVCore::Category::PROJECTS: {
		Q_EMIT load_project_from_path(var.toString());
		break;
	}
	case PVCore::Category::SOURCES: {
		PVCore::PVSerializedSource ss = var.value<PVCore::PVSerializedSource>();
		if (ss.need_credential()) {
			PVGuiQt::CredentialDialog dial;
			if (dial.exec() != QDialog::Accepted) {
				break;
			}
			for (std::vector<std::string>& p : ss.input_desc) {
				p.push_back(dial.get_login().toStdString());
				p.push_back(dial.get_password().toStdString());
			}
		}
		try {
			Q_EMIT load_source_from_description(PVRush::PVSourceDescription(ss));
		} catch (std::runtime_error& e) {
			PVLOG_ERROR(e.what());
			refresh_all_recent_items();
		}
		break;
	}
	case PVCore::Category::EDITED_FORMATS:
	case PVCore::Category::USED_FORMATS: {
		Q_EMIT edit_format(var.toString());
		break;
	}
	default: {
		break;
	}
	}
}

size_t PVGuiQt::PVStartScreenWidget::selected_count(PVCore::Category cat)
{
	custom_listwidget_t* list = _recent_list_widgets[cat];

	size_t selected_cout = 0;
	for (int i = 0; i < list->count(); i++) {
		__impl::PVListWidgetItem* item = (__impl::PVListWidgetItem*)list->item(i);
		assert(item);
		selected_cout += item->is_checked();
	}

	return selected_cout;
}

size_t PVGuiQt::PVStartScreenWidget::total_count(PVCore::Category cat)
{
	custom_listwidget_t* list = _recent_list_widgets[cat];
	return list->count();
}

void PVGuiQt::PVStartScreenWidget::delete_investigation_dlg()
{
	auto dlg = new __impl::PVDeleteInvestigationDialog(this);

	constexpr PVCore::Category cat = PVCore::Category::PROJECTS;

	custom_listwidget_t* list = _recent_list_widgets[cat];

	if (dlg->exec() == QDialog::Accepted) {
		for (int i = 0; i < list->count(); i++) {
			__impl::PVListWidgetItem* item = (__impl::PVListWidgetItem*)list->item(i);
			assert(item);

			QVariant var = item->data(Qt::UserRole);
			QString filename = var.toString();

			if (item->is_checked()) {
				if (dlg->remove_cache() || dlg->delete_investigation()) {
					PVRush::PVNrawCacheManager::get().remove_investigation(
					    filename, dlg->delete_investigation());
				}
			}
		}

		if (dlg->clear_history() || dlg->delete_investigation()) {
			clear_history<cat>();
		}
	}
}

QString PVGuiQt::PVStartScreenWidget::format_selected_item_string(PVCore::Category cat)
{
	size_t sel_count = selected_count(cat);
	size_t tot_count = total_count(cat);

	QString c;
	if ((sel_count == 0 && tot_count > 1)) {
		c = QString::number(tot_count) + " selected items";
	} else if (sel_count > 1) {
		c = QString::number(sel_count) + " selected items";
	} else {
		c = "selected item";
	}

	return c;
}

/******************************************************************************
 *
 * PVGuiQt::PVStartScreenWidget::PVListWidgetItem
 *
 *****************************************************************************/
PVGuiQt::__impl::PVListWidgetItem::PVListWidgetItem(
    PVCore::Category cat,
    QString long_string,
    QStringList filenames,
    QVariant var,
    int index,
    PVGuiQt::PVStartScreenWidget::custom_listwidget_t* parent,
    PVGuiQt::PVStartScreenWidget* start_screen_widget)
    : QListWidgetItem(parent), _cat(cat)
{
	setData(Qt::UserRole, var);
	setData(Qt::UserRole + 1, cat);
	QString short_string;
	switch (cat) {
	case PVCore::Category::SOURCES: {
		size_t brac_open_pos = long_string.indexOf("[");
		size_t brac_close_pos = long_string.indexOf("]");
		QString format_name = long_string.mid(brac_open_pos, brac_close_pos);
		if (filenames.size() == 1) {
			short_string =
			    PVWidgets::PVUtils::shorten_path(long_string.left(brac_open_pos),
			                                     *PVGuiQt::PVStartScreenWidget::_item_font,
			                                     PVGuiQt::PVStartScreenWidget::_item_width) +
			    format_name;
		} else {
			short_string = "Aggregation " + format_name;
		}
		break;
	}
	default:
		short_string =
		    PVWidgets::PVUtils::shorten_path(long_string, *PVGuiQt::PVStartScreenWidget::_item_font,
		                                     PVGuiQt::PVStartScreenWidget::_item_width);
		break;
	}

	auto layout = new QHBoxLayout();
	layout->setAlignment(Qt::AlignLeft);
	_widget = new QWidget();
	_widget->setLayout(layout);

	// Icon
	_icon_label = new QLabel();
	QIcon icon;
	if (filenames.size() == 1) {
		QFileInfo finfo(filenames[0]);
		QFileIconProvider ficon;
		icon = ficon.icon(finfo);
	} else {
		icon = QApplication::style()->standardIcon(QStyle::SP_FileDialogNewFolder);
	}
	_icon_label->setPixmap(icon.pixmap(15, 15));
	_icon_label->setMouseTracking(true);
	_icon_label->installEventFilter(this);
	_checkbox = new QCheckBox();
	_checkbox->setMouseTracking(true);
	_checkbox->installEventFilter(this);
	_checkbox->setVisible(false);
	layout->addWidget(_checkbox);
	layout->addWidget(_icon_label);

	// Text
	auto text_label = new QLabel();
	text_label->setTextFormat(Qt::RichText);
	text_label->setText(QString("<a href=\"%1;%2\">" + short_string + "</a>").arg(cat).arg(index));
	text_label->setToolTip(long_string);
	connect(text_label, &QLabel::linkActivated, start_screen_widget,
	        &PVStartScreenWidget::dispatch_action);
	layout->addWidget(text_label);

	setSizeHint(QSize(_widget->sizeHint().width(),
	                  _widget->sizeHint().height() - 6)); // Do not forget this!

	// This ugly workaround is needed to avoid missing QEvent::Leave events when
	// switch from checkbox to icon
	_timer.setSingleShot(true);
	_timer.setInterval(50);
	connect(&_timer, &QTimer::timeout, this, &PVListWidgetItem::timeout);
}

void PVGuiQt::__impl::PVListWidgetItem::timeout()
{
	if (!_checkbox->rect().contains(_checkbox->mapFromGlobal(QCursor::pos()))) {
		_icon_label->setVisible(true);
		_checkbox->setVisible(false);
	}
}

bool PVGuiQt::__impl::PVListWidgetItem::eventFilter(QObject* obj, QEvent* event)
{
	if (obj == _icon_label) {
		if (event->type() == QEvent::Enter) {
			set_icon_visible(false);
			return true;
		}
	} else if (obj == _checkbox) {
		if (event->type() == QEvent::Leave) {
			if (obj == _checkbox && !_checkbox->isChecked()) {
				set_icon_visible(true);
				return true;
			}
		} else if (event->type() == QEvent::Show) {
			_timer.start();
		}
	}

	return false;
}

void PVGuiQt::__impl::PVListWidgetItem::set_icon_visible(bool visible)
{
	_checkbox->setChecked(false);
	_icon_label->setVisible(visible);
	_checkbox->setVisible(!visible);
}

/******************************************************************************
 *
 * PVGuiQt::PVStartScreenWidget::PVDeleteInvestigationDialog
 *
 *****************************************************************************/
PVGuiQt::__impl::PVDeleteInvestigationDialog::PVDeleteInvestigationDialog(
    PVStartScreenWidget* parent)
    : QDialog(parent)
{
	auto vbox = new QVBoxLayout();
	_clear_history_cb = new QCheckBox("Clear from history");
	_remove_cache_cb = new QCheckBox("Clear import cache");
	_delete_investigation_cb = new QCheckBox("Delete investigation");
	connect(_delete_investigation_cb, &QCheckBox::stateChanged, this,
	        &PVDeleteInvestigationDialog::delete_investigation_checked);

	auto button_box = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	connect(button_box, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(button_box, &QDialogButtonBox::rejected, this, &QDialog::reject);

	QString c = start_screen_widget()->format_selected_item_string(PVCore::Category::PROJECTS);

	vbox->addWidget(new QLabel("Choose the action to perform on the " + c + ":\n"));
	vbox->addWidget(_clear_history_cb);
	vbox->addWidget(_remove_cache_cb);
	vbox->addWidget(_delete_investigation_cb);
	vbox->addWidget(button_box);

	setLayout(vbox);
}

void PVGuiQt::__impl::PVDeleteInvestigationDialog::delete_investigation_checked(int state)
{
	if (state == Qt::Checked) {
		_old_clear_history_state = _clear_history_cb->isChecked();
		_old_remove_cache_state = _remove_cache_cb->isChecked();
		_clear_history_cb->setChecked(true);
		_clear_history_cb->setEnabled(false);
		_remove_cache_cb->setChecked(true);
		_remove_cache_cb->setEnabled(false);
	} else {
		_clear_history_cb->setChecked(_old_clear_history_state);
		_clear_history_cb->setEnabled(true);
		_remove_cache_cb->setChecked(_old_remove_cache_state);
		_remove_cache_cb->setEnabled(true);
	}
}

namespace PVGuiQt
{

template <>
void PVStartScreenWidget::refresh_recent_items<PVCore::Category::SOURCES>()
{
	custom_listwidget_t* list = _recent_list_widgets[PVCore::Category::SOURCES];
	QPushButton* clear_button = _recent_push_buttons[PVCore::Category::SOURCES];
	list->setObjectName("RecentProjectItem");
	list->clear();

	PVCore::PVRecentItemsManager::get().clear_missing_files();

	uint64_t index = 0;
	for (auto const& sd :
	     PVCore::PVRecentItemsManager::get().get_list<PVCore::Category::SOURCES>()) {
		// item + data
		QString long_string;
		QStringList filenames;
		std::tie(long_string, filenames) =
		    PVCore::PVRecentItemsManager::get().get_string_from_entry(sd);

		QVariant var;
		var.setValue(PVCore::PVSerializedSource(sd));
		__impl::PVListWidgetItem* item_widget = new __impl::PVListWidgetItem(
		    PVCore::Category::SOURCES, long_string, filenames, var, index, list, this);
		list->setItemWidget(item_widget, item_widget->widget());

		index++;
	}

	if (clear_button) {
		clear_button->setEnabled(index > 0);
	}
}

} // namespace PVGuiQt
