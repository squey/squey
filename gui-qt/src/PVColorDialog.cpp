/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/general.h>
#include <PVColorDialog.h>


/******************************************************************************
 *
 * PVInspector::PVColorDialog::PVColorDialog
 *
 *****************************************************************************/
PVInspector::PVColorDialog::PVColorDialog(Inendi::PVView& inendi_view, QWidget* parent):
	QColorDialog(Qt::white, parent),
	_inendi_view(inendi_view)
{
	setOption(QColorDialog::ShowAlphaChannel, true);
	//setWindowFlags(Qt::WindowStaysOnTopHint);
	setWindowTitle("Select a color...");
}
