//! \file PVColorDialog.h
//! $Id: PVColorDialog.h 2498 2011-04-25 14:27:23Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCOLORDIALOG_H
#define PVCOLORDIALOG_H

#include <QColorDialog>

namespace PVInspector {
class PVMainWindow;

/**
 * \class PVColorDialog
 */
class PVColorDialog : public QColorDialog
{
Q_OBJECT

	PVMainWindow *main_window;

public:
	/**
	 * Constructor
	 */
	PVColorDialog(PVMainWindow *mw, QWidget *parent = 0);
};
}

#endif // PVCOLORDIALOG_H

