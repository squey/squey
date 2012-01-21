//! \file PVStartScreenWidget.h
//! $Id: PVStartScreenWidget.h 2496 2011-04-25 14:10:00Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2012
//! Copyright (C) Philippe Saadé 2009-2012
//! Copyright (C) Picviz Labs 2012

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
		QWidget          *format_widget;
		QWidget          *import_widget;
		QWidget          *investigation_widget;

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


