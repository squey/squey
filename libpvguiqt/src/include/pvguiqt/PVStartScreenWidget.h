/**
 * \file PVStartScreenWidget.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVSTARTSCREENWIDGET_H
#define PVSTARTSCREENWIDGET_H

#include <tuple>

#include <QWidget>
#include <QStringList>
#include <QVBoxLayout>
#include <QListWidgetItem>
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
 *  \class PVStartScreenWidget
 */
class PVStartScreenWidget : public QWidget
{
	Q_OBJECT
public:
	typedef std::tuple<QString, QString, QStringList> descr_strings_t;
	//typedef PVWidgets::PVSizeHintListWidget<QListWidget, 42> custom_listwidget_t;
	typedef QListWidget custom_listwidget_t;

public:
	PVStartScreenWidget(QWidget* parent = 0);
	void refresh_all_recent_items();
	void refresh_recent_sources_items();
	void refresh_recent_items(int category);

signals:
	void new_project();
	void load_project();
	void load_project_from_path(const QString & project);
	void load_source_from_description(PVRush::PVSourceDescription);
	void new_format();
	void load_format();
	void edit_format(const QString & project);
	void import_type(const QString &);

public slots:
	void dispatch_action(const QString& id);
	void import_type_Slot();

private:
	descr_strings_t get_string_from_variant(PVCore::PVRecentItemsManager::Category category, const QVariant& var);
	descr_strings_t get_string_from_format(const QVariant& var);
	descr_strings_t get_string_from_source_description(const QVariant& var);

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


