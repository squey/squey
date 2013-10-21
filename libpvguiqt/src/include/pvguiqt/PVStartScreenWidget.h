/**
 * \file PVStartScreenWidget.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVSTARTSCREENWIDGET_H
#define PVSTARTSCREENWIDGET_H

#include <tuple>

#include <QDialog>
#include <QStringList>
#include <QListWidget>
#include <QCheckBox>
#include <QTimer>
class QLabel;

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVObserverSignal.h>

#include <pvkernel/widgets/PVSizeHintListWidget.h>
#include <pvkernel/core/PVRecentItemsManager.h>

namespace PVGuiQt {

namespace __impl {
class PVListWidgetItem;
class PVDeleteInvestigationDialog;
}

class PVStartScreenWidget;

class PVAddRecentItemFuncObserver: public PVHive::PVFuncObserverSignal<PVCore::PVRecentItemsManager, FUNC(PVCore::PVRecentItemsManager::add)>
{
public:
	PVAddRecentItemFuncObserver(PVStartScreenWidget* parent) : _parent(parent) {}
public:
	void update(const arguments_deep_copy_type& args) const;
private:
	PVStartScreenWidget* _parent;
};

class PVAddSourceRecentItemFuncObserver: public PVHive::PVFuncObserverSignal<PVCore::PVRecentItemsManager, FUNC(PVCore::PVRecentItemsManager::add_source)>
{
public:
	PVAddSourceRecentItemFuncObserver(PVStartScreenWidget* parent) : _parent(parent) {}
public:
	void update(const arguments_deep_copy_type& args) const;
private:
	PVStartScreenWidget* _parent;
};

/**
 * \class PVRecentItemsManager
 *
 * \note This class is the start screen widget of the application.
 *       It displays the recent items accessed by the user/application and allow to load/edit them.
 */
class PVStartScreenWidget : public QWidget
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
	void refresh_recent_items(int category);

	QString format_selected_item_string(PVCore::PVRecentItemsManager::Category cat);

signals:
	// These signals are used by to the PVMainWindow.
	void new_project();
	void load_project();
	void load_project_from_path(const QString & project);
	void load_source_from_description(PVRush::PVSourceDescription);
	void new_format();
	void load_format();
	void edit_format(const QString & project);
	void import_type(const QString &);

public slots:
	/*! \brief Slot called when clicking on the hyperlink of a recent item and emitting the proper signal.
	 *  \param[in] id The item identifier under the following format: "category_index;item_index"
	 */
	void dispatch_action(const QString& id);

	/*! \brief Slot emitting the proper "import_type" signal.
	 *  \note used by PVGuiQt::PVInputTypeMenuEntries::add_inputs_to_layout
	 */
	void import_type();

private:
	/*! \brief Extract a descr_strings_t from a QVariant of a given category.
	 *  \param[in] category The given category.
	 *  \param[in] var      The QVariant to convert to descr_strings_t.
	 */
	static descr_strings_t get_strings_from_variant(PVCore::PVRecentItemsManager::Category category, const QVariant& var);

	/*! \brief Extract a descr_strings_t from a Picviz::PVFormat QVariant.
	 *  \param[in] var The Picviz::PVFormat QVariant to convert to descr_strings_t.
	 */
	static descr_strings_t get_strings_from_format(const QVariant& var);

	/*! \brief Extract a descr_strings_t from a PVRush::PVSourceDescription QVariant.
	 *  \param[in] var The PVRush::PVSourceDescription QVariant to convert to descr_strings_t.
	 */
	static descr_strings_t get_strings_from_source_description(const QVariant& var);

	void clear_history(PVCore::PVRecentItemsManager::Category cat);

	void clear_history_dlg(PVCore::PVRecentItemsManager::Category cat);
	void delete_investigation_dlg();

	size_t selected_count(PVCore::PVRecentItemsManager::Category cat);
	size_t total_count(PVCore::PVRecentItemsManager::Category cat);


private:
	QWidget* format_widget;
	QWidget* import_widget;
	QWidget* project_widget;

	custom_listwidget_t* _recent_list_widgets[PVCore::PVRecentItemsManager::Category::LAST];
	QPushButton* _recent_push_buttons[PVCore::PVRecentItemsManager::Category::LAST];

	PVAddRecentItemFuncObserver _recent_items_add_obs;
	PVAddSourceRecentItemFuncObserver _recent_items_add_source_obs;

	static QFont _item_font;
	static const uint64_t _item_width = 475;
};

namespace __impl
{

class PVListWidgetItem : public QObject, public QListWidgetItem
{
	Q_OBJECT

public:
	PVListWidgetItem(
		PVCore::PVRecentItemsManager::Category cat,
		QVariant var,
		int index,
		PVGuiQt::PVStartScreenWidget::custom_listwidget_t* parent,
		PVGuiQt::PVStartScreenWidget* start_screen_widget
	);

protected:
	bool eventFilter(QObject* obj, QEvent* event) override;

public:
	QWidget* widget() { return _widget; }
	bool is_checked() { return _checkbox->isChecked(); }
	void set_icon_visible(bool visible);

private slots:
	void timeout();

private:
	QCheckBox* _checkbox;
	QLabel* _icon_label;
	QWidget* _widget;
	PVCore::PVRecentItemsManager::Category _cat;
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
	PVStartScreenWidget* start_screen_widget() { return static_cast<PVStartScreenWidget*>(parent()); }

private slots:
 	void delete_investigation_checked(int state);

private:
	QCheckBox* _clear_history_cb;
	QCheckBox* _remove_cache_cb;
	QCheckBox* _delete_investigation_cb;
	bool _old_clear_history_state;
	bool _old_remove_cache_state;
};

}

}

#endif // PVSTARTSCREENWIDGET_H


