/**
 * \file PVColorDialog.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

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
