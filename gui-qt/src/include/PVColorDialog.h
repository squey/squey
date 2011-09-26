//! \file PVColorDialog.h
//! $Id: PVColorDialog.h 2498 2011-04-25 14:27:23Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

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

