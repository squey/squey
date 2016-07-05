/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVPlotted.h>
#include <inendi/widgets/PVMappingPlottingEditDialog.h>
#include <pvguiqt/PVQPlotted.h>

bool PVGuiQt::PVQPlotted::edit_plotted(Inendi::PVPlotted& plotted, QWidget* parent)
{
	PVWidgets::PVMappingPlottingEditDialog* dlg =
	    new PVWidgets::PVMappingPlottingEditDialog(NULL, &plotted.get_plotting(), parent);
	if (dlg->exec() != QDialog::Accepted) {
		return false;
	}

	plotted.update_plotting();

	return true;
}
