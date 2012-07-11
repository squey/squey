//! \file PVColorDialog.cpp
//! $Id: PVColorDialog.cpp 2496 2011-04-25 14:10:00Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvkernel/core/general.h>
#include <PVColorDialog.h>


/******************************************************************************
 *
 * PVInspector::PVColorDialog::PVColorDialog
 *
 *****************************************************************************/
PVInspector::PVColorDialog::PVColorDialog(Picviz::PVView& picviz_view, QWidget* parent):
	QColorDialog(Qt::white, parent),
	_picviz_view(picviz_view)
{
	setOption(QColorDialog::ShowAlphaChannel, true);
	//setWindowFlags(Qt::WindowStaysOnTopHint);
	setWindowTitle("Select a color...");
}
