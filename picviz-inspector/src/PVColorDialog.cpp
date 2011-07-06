//! \file PVColorDialog.cpp
//! $Id: PVColorDialog.cpp 2496 2011-04-25 14:10:00Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtGui>

#include <pvcore/general.h>

#include <PVMainWindow.h>

#include <PVColorDialog.h>

// FIXME! Is this even used?


/******************************************************************************
 *
 * PVInspector::PVColorDialog::PVColorDialog
 *
 *****************************************************************************/
PVInspector::PVColorDialog::PVColorDialog(PVMainWindow *mw, QWidget *parent) : QColorDialog(parent)
{
	PVLOG_DEBUG("PVColorDialog::%s\n", __FUNCTION__);

	main_window = mw;

	setWindowTitle("Color dialog"); // XXX this should probably be marked for translation.

}
