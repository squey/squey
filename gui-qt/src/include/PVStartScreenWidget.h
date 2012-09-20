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
		void update_recent_projects();

	private:
		PVMainWindow* _mw;

		QVBoxLayout* _recent_projects_layout;

		QWidget* format_widget;
		QWidget* import_widget;
		QWidget* project_widget;
};
}

#endif // PVSTARTSCREENWIDGET_H


