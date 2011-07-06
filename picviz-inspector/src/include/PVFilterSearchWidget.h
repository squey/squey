//! \file Pvfiltersearchwidget.h
//! $Id: PVFilterSearchWidget.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTERSEARCHWIDGET_H
#define PVFILTERSEARCHWIDGET_H

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
 * \class PVFilterSearchWidget
 */
class PVFilterSearchWidget : public QDialog
{
	Q_OBJECT

	/* QT Widgets */
	QGridLayout *filter_widgets_layout;  //!<

	/* libpicviz objects */
	// FIXME!picviz_filter_t *filter;
	char            *filter_name;        //!<
	PVMainWindow    *main_window;        //!<

public:

	QLineEdit *searchline_edit;
	QComboBox *combo_axes_list;

	/**
	 * Constructor.
	 */
	PVFilterSearchWidget(PVMainWindow *parent, QString const& regexp, int axis_id, bool show_regexp = true);

	int _axis_id;
	QString _regexp;
public slots:
	/**
	 *
	 */
	void search_filter_cancel_action_Slot();


	/**
	 *
	 */
	void search_filter_ok_action_Slot();


signals:
	/**
	 *
	 */
	void search_filter_applied_Signal();
};
}

#endif // PVFILTERSEARCHWIDGET_H



