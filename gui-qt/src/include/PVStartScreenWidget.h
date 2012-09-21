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

#include <PVMainWindow.h>

namespace PVInspector {

/**
 *  \class PVStartScreenWidget
 */
class PVStartScreenWidget : public QWidget
{
	Q_OBJECT

	public:
		PVStartScreenWidget(PVMainWindow* parent);
		void update_recent_items(PVMainWindow::ERecentItemsCategory category);
		void update_all_recent_items();

	private:
		PVMainWindow* _mw;

		//QVBoxLayout* _recent_layouts[4];

		QWidget* format_widget;
		QWidget* import_widget;
		QWidget* project_widget;

		struct PVRecentList {
			PVRecentList(QVBoxLayout* l = nullptr, const char* s = nullptr) : layout(l), slot(s) {}
			QVBoxLayout* layout;
			const char* slot;
		};

		PVRecentList _recent_lists[4];
};
}

#endif // PVSTARTSCREENWIDGET_H


