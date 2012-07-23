/**
 * \file PVStartScreenWidget.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVSTARTSCREENWIDGET_H
#define PVSTARTSCREENWIDGET_H

#include <QAction>
#include <QWidget>


namespace PVInspector {
class PVMainWindow;

/**
 *  \class PVStartScreenWidget
 */
class PVStartScreenWidget : public QWidget
{
	Q_OBJECT

		PVMainWindow     *main_window;            //!<

	public:
		/**
		 * Constructor.
		 */
		PVStartScreenWidget(PVMainWindow *mw, PVMainWindow *parent);

		/**
		 *
		 * @return
		 */

	public slots:
//		void show_hide_layer_stack_Slot(bool visible);

	public:
		//void refresh();
};
}

#endif // PVSTARTSCREENWIDGET_H


