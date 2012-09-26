/**
 * \file PVStartScreenWidget.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVSTARTSCREENWIDGET_H
#define PVSTARTSCREENWIDGET_H

#include <QWidget>
#include <QStringList>
#include <QVBoxLayout>
#include <QListWidgetItem>
#include <QListWidget>

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVObserverSignal.h>

#include <pvkernel/core/PVRecentItemsManager.h>

namespace PVInspector {
class PVMainWindow;
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
		PVStartScreenWidget(PVMainWindow* parent);
		void refresh_all_recent_items();
		void refresh_recent_sources_items();

		void refresh_recent_items(int category);

	public slots:
		void dispatch_action(const QString& id);

	private:
		QString get_string_from_variant(PVCore::PVRecentItemsManager::Category category, const QVariant& var);
		QString get_string_from_format(const QVariant& var);
		QString get_string_from_source_description(const QVariant& var);

	private:
		PVMainWindow* _mw;

		QWidget* format_widget;
		QWidget* import_widget;
		QWidget* project_widget;

		QListWidget* _recent_list_widgets[PVCore::PVRecentItemsManager::Category::LAST];

		PVAddRecentItemFuncObserver _recent_items_add_obs;
		PVAddSourceRecentItemFuncObserver _recent_items_add_source_obs;
};
}

#endif // PVSTARTSCREENWIDGET_H


