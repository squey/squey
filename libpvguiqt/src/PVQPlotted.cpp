/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVPlotted.h>
#include <inendi/widgets/PVMappingPlottingEditDialog.h>
#include <pvguiqt/PVQPlotted.h>

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVHive.h>

bool PVGuiQt::PVQPlotted::edit_plotted(Inendi::PVPlotted& plotted, QWidget* parent)
{
	PVWidgets::PVMappingPlottingEditDialog* dlg =
	    new PVWidgets::PVMappingPlottingEditDialog(NULL, &plotted.get_plotting(), parent);
	if (dlg->exec() != QDialog::Accepted) {
		return false;
	}

	if (plotted.is_current_plotted()) {
		Inendi::PVPlotted_sp plotted_sp = plotted.shared_from_this();
		PVHive::call<FUNC(Inendi::PVPlotted::plotting_updated)>(plotted_sp);
	}

	return true;
}
