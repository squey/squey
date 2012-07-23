/**
 * \file PVColorDialog.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVCOLORDIALOG_H
#define PVCOLORDIALOG_H

#include <QColorDialog>
#include <picviz/PVView.h>

namespace PVInspector {
class PVMainWindow;

/**
 * \class PVColorDialog
 */
class PVColorDialog : public QColorDialog
{
	Q_OBJECT

public:
	/**
	 * Constructor
	 */
	PVColorDialog(Picviz::PVView_p picviz_view, QWidget* parent = 0);

public:
	Picviz::PVView_p get_lib_view() { return _picviz_view; }

protected:
	Picviz::PVView_p _picviz_view;
};
}

#endif // PVCOLORDIALOG_H

