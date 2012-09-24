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

/**
 *  \class PVStartScreenWidget
 */
class PVStartScreenWidget : public QWidget
{
	Q_OBJECT

	public:
		PVStartScreenWidget(PVMainWindow* parent);
		void refresh_all_recent_items();

		void refresh_recent_items(int category);

	private:
		PVMainWindow* _mw;

		QWidget* format_widget;
		QWidget* import_widget;
		QWidget* project_widget;

		struct PVRecentList {
			PVRecentList(QVBoxLayout* l = nullptr, const char* s = nullptr) : layout(l), slot(s) {}
			QVBoxLayout* layout;
			const char* slot;
		};

		PVRecentList _recent_lists[PVCore::PVRecentItemsManager::Category::LAST];

		PVAddRecentItemFuncObserver _recent_items_add_obs;
};
}

#endif // PVSTARTSCREENWIDGET_H


