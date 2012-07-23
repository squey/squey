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
PVInspector::PVColorDialog::PVColorDialog(Picviz::PVView_p picviz_view, QWidget* parent):
	QColorDialog(Qt::white, parent),
	_picviz_view(picviz_view)
{
	PVLOG_DEBUG("PVColorDialog::%s\n", __FUNCTION__);

	setOption(QColorDialog::ShowAlphaChannel, true);
	//setWindowFlags(Qt::WindowStaysOnTopHint);
	setWindowTitle("Select a color...");
}
