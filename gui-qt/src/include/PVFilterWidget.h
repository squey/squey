//! \file PVFilterWidget.h
//! $Id: PVFilterWidget.h 2496 2011-04-25 14:10:00Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTERWIDGET_H
#define PVFILTERWIDGET_H

#include <QtCore>
#include <QDialog>
#include <QGridLayout>

#include <PVTabSplitter.h>

#include <picviz/general.h>
#include <picviz/arguments.h>
//#include <picviz/filters.h>

namespace PVInspector {
class PVMainWindow;

/**
 * \class PVFilterWidget
 */
class PVFilterWidget : public QDialog
{
	Q_OBJECT

	/* QT Widgets */
	QGridLayout *filter_widgets_layout;  //!<

	/* libpicviz objects */
	// FIXME!picviz_filter_t *filter;

	PVTabSplitter   *current_tab;        //!<
	char            *filter_name;        //!<

	PVMainWindow    *main_window;        //!<

public:
	/**
	 * Constructor.
	 */
	PVFilterWidget(PVMainWindow *parent);

	/**
	 *
	 * @param filter_name
	 * @param filter
	 */
	// FIXME!void create(QString filter_name = NULL, picviz_filter_t *filter = NULL);

	/**
	 *
	 * @return
	 */
	picviz_arguments_t *filter_exec_action_build_arguments(void);

public slots:
	/**
	 *
	 */
	void filter_cancel_action_Slot();

	/**
	 *
	 */
	void filter_apply_action_Slot();

	/**
	 *
	 */
	void filter_ok_action_Slot();

signals:
	/**
	 *
	 */
	void filter_applied_Signal();
};
}

#endif // PVFILTERWIDGET_H



