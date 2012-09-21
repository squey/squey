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

#include <pvguiqt/PVRecentItemsManager.h>

namespace PVInspector {
class PVMainWindow;

/**
 *  \class PVStartScreenWidget
 */
class PVStartScreenWidget : public QWidget
{
	Q_OBJECT

	public:
		PVStartScreenWidget(PVMainWindow* parent);
		void refresh_all_recent_items();

	public slots:
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

		PVRecentList _recent_lists[PVGuiQt::PVRecentItemsManager::Category::LAST];
};
}

#endif // PVSTARTSCREENWIDGET_H


