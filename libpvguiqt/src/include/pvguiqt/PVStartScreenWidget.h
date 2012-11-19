/**
 * \file PVStartScreenWidget.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVSTARTSCREENWIDGET_H
#define PVSTARTSCREENWIDGET_H

#include <tuple>

#include <QStringList>
#include <QListWidget>

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVObserverSignal.h>

#include <pvkernel/widgets/PVSizeHintListWidget.h>
#include <pvkernel/core/PVRecentItemsManager.h>

namespace PVGuiQt {

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
	descr_strings_t get_strings_from_variant(PVCore::PVRecentItemsManager::Category category, const QVariant& var);

	/*! \brief Extract a descr_strings_t from a Picviz::PVFormat QVariant.
	 *  \param[in] var The Picviz::PVFormat QVariant to convert to descr_strings_t.
	 */
	descr_strings_t get_strings_from_format(const QVariant& var);

	/*! \brief Extract a descr_strings_t from a PVRush::PVSourceDescription QVariant.
	 *  \param[in] var The PVRush::PVSourceDescription QVariant to convert to descr_strings_t.
	 */
	descr_strings_t get_strings_from_source_description(const QVariant& var);

private:
	QWidget* format_widget;
	QWidget* import_widget;
	QWidget* project_widget;

	custom_listwidget_t* _recent_list_widgets[PVCore::PVRecentItemsManager::Category::LAST];

	PVAddRecentItemFuncObserver _recent_items_add_obs;
	PVAddSourceRecentItemFuncObserver _recent_items_add_source_obs;

	QFont _item_font;
	uint64_t _item_width = 475;
};
}

#endif // PVSTARTSCREENWIDGET_H


