/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVSTARTSCREENWIDGET_H
#define PVSTARTSCREENWIDGET_H

#include <sigc++/sigc++.h>

#include <tuple>

#include <QDialog>
#include <QStringList>
#include <QListWidget>
#include <QCheckBox>
#include <QTimer>
#include <QPushButton>
#include <QMessageBox>
class QLabel;

#include <pvkernel/core/PVRecentItemsManager.h>

namespace PVGuiQt
{

namespace __impl
{
class PVListWidgetItem;
class PVDeleteInvestigationDialog;
}

class PVStartScreenWidget;

/**
 * \class PVRecentItemsManager
 *
 * \note This class is the start screen widget of the application.
 *       It displays the recent items accessed by the user/application and allow to load/edit them.
 */
class PVStartScreenWidget : public QWidget, public sigc::trackable
{
	Q_OBJECT

	friend class __impl::PVListWidgetItem;

  public:
	// Store the description strings under a 3-tuple of: short_string, long_string, filenames
	typedef std::tuple<QString, QString, QStringList> descr_strings_t;
	typedef QListWidget custom_listwidget_t;

  public:
	PVStartScreenWidget(QWidget* parent = 0);

  public:
	/*! \brief Refresh the recent items of all categories.
	 */
	void refresh_all_recent_items();

	/*! \brief Refresh the recent sources items.
	 */
	void refresh_recent_sources_items();

	/*! \brief Refresh the recent items of a given category.
	*/
	template <PVCore::Category category>
	void refresh_recent_items();

	QString format_selected_item_string(PVCore::Category cat);

  Q_SIGNALS:
	// These signals are used by to the PVMainWindow.
	void new_project();
	void load_project();
	void load_project_from_path(const QString& project);
	void load_source_from_description(PVRush::PVSourceDescription);
	void new_format();
	void load_format();
	void edit_format(const QString& project);
	void import_type(const QString&);

  public Q_SLOTS:
	/*! \brief Slot called when clicking on the hyperlink of a recent item and emitting the proper
	 * signal.
	 *  \param[in] id The item identifier under the following format: "category_index;item_index"
	 */
	void dispatch_action(const QString& id);

	/*! \brief Slot emitting the proper "import_type" signal.
	 *  \note used by PVGuiQt::PVInputTypeMenuEntries::add_inputs_to_layout
	 */
	void import_type();

  private:
	template <PVCore::Category cat>
	void clear_history();

	template <PVCore::Category cat>
	void clear_history_dlg();

	void delete_investigation_dlg();

	size_t selected_count(PVCore::Category cat);
	size_t total_count(PVCore::Category cat);

  private:
	QWidget* format_widget;
	QWidget* import_widget;
	QWidget* project_widget;

	custom_listwidget_t* _recent_list_widgets[PVCore::Category::LAST];
	QPushButton* _recent_push_buttons[PVCore::Category::LAST];

	static const QFont* _item_font;
	static const uint64_t _item_width = 475;
};

namespace __impl
{

class PVListWidgetItem : public QObject, public QListWidgetItem
{
	Q_OBJECT

  public:
	PVListWidgetItem(PVCore::Category cat,
	                 QString long_string,
	                 QStringList filenames,
	                 QVariant var,
	                 int index,
	                 PVGuiQt::PVStartScreenWidget::custom_listwidget_t* parent,
	                 PVGuiQt::PVStartScreenWidget* start_screen_widget);

  protected:
	bool eventFilter(QObject* obj, QEvent* event) override;

  public:
	QWidget* widget() { return _widget; }
	bool is_checked() { return _checkbox->isChecked(); }
	void set_icon_visible(bool visible);

  private Q_SLOTS:
	void timeout();

  private:
	QCheckBox* _checkbox;
	QLabel* _icon_label;
	QWidget* _widget;
	PVCore::Category _cat;
	QTimer _timer;
};

class PVDeleteInvestigationDialog : public QDialog
{
	Q_OBJECT

  public:
	PVDeleteInvestigationDialog(PVStartScreenWidget* parent);

  public:
	bool clear_history() { return _clear_history_cb->isChecked(); }
	bool remove_cache() { return _remove_cache_cb->isChecked(); }
	bool delete_investigation() { return _delete_investigation_cb->isChecked(); }

  private:
	PVStartScreenWidget* start_screen_widget()
	{
		return static_cast<PVStartScreenWidget*>(parent());
	}

  private Q_SLOTS:
	void delete_investigation_checked(int state);

  private:
	QCheckBox* _clear_history_cb;
	QCheckBox* _remove_cache_cb;
	QCheckBox* _delete_investigation_cb;
	bool _old_clear_history_state;
	bool _old_remove_cache_state;
};
}

template <PVCore::Category category>
void PVStartScreenWidget::refresh_recent_items()
{
	custom_listwidget_t* list = _recent_list_widgets[category];
	QPushButton* clear_button = _recent_push_buttons[category];
	list->setObjectName("RecentProjectItem");
	list->clear();

	uint64_t index = 0;
	for (QString var : PVCore::PVRecentItemsManager::get().get_list<category>()) {
		// item + data
		QString long_string;
		QStringList filenames;
		std::tie(long_string, filenames) =
		    PVCore::PVRecentItemsManager::get().get_string_from_entry(var);
		__impl::PVListWidgetItem* item_widget =
		    new __impl::PVListWidgetItem(category, long_string, filenames, var, index, list, this);
		list->setItemWidget(item_widget, item_widget->widget());

		index++;
	}

	if (clear_button) {
		clear_button->setEnabled(index > 0);
	}
}

template <PVCore::Category category>
void PVStartScreenWidget::clear_history()
{
	custom_listwidget_t* list = _recent_list_widgets[category];
	QList<int> indexes;

	for (int i = list->count(); i-- > 0;) {
		__impl::PVListWidgetItem* item = (__impl::PVListWidgetItem*)list->item(i);
		assert(item);
		if (item->is_checked()) {
			indexes << i;
		}
	}

	// Clear list widget
	if (indexes.isEmpty()) {
		list->clear();
	}

	// Clear config file
	PVCore::PVRecentItemsManager::get().clear(category, indexes);

	refresh_recent_items<category>();
}

template <PVCore::Category category>
void PVStartScreenWidget::clear_history_dlg()
{
	QString c = format_selected_item_string(category);
	QMessageBox confirm(QMessageBox::Question, tr("Please confirm"),
	                    "Clear history for the " + c + "?", QMessageBox::Yes | QMessageBox::No,
	                    this);
	if (confirm.exec() == QMessageBox::Yes) {
		clear_history<category>();
	}
}
}

#endif // PVSTARTSCREENWIDGET_H
