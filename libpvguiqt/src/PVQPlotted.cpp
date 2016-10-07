/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVPlotted.h>
#include <inendi/widgets/PVMappingPlottingEditDialog.h>
#include <pvguiqt/PVQPlotted.h>
#include <pvkernel/core/PVProgressBox.h>

bool PVGuiQt::PVQPlotted::edit_plotted(Inendi::PVPlotted& plotted, QWidget* parent)
{
	PVWidgets::PVMappingPlottingEditDialog* dlg =
	    new PVWidgets::PVMappingPlottingEditDialog(nullptr, &plotted, parent);
	if (dlg->exec() != QDialog::Accepted) {
		return false;
	}

	PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& /*pbox*/) { plotted.update_plotting(); },
	    QObject::tr("Updating plotting(s)..."), nullptr);

	return true;
}
