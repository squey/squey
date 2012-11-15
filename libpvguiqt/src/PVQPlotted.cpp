#include <picviz/PVPlotted.h>
#include <picviz/widgets/PVMappingPlottingEditDialog.h>
#include <pvguiqt/PVQPlotted.h>

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVHive.h>

bool PVGuiQt::PVQPlotted::edit_plotted(Picviz::PVPlotted& plotted, QWidget* parent)
{
	PVWidgets::PVMappingPlottingEditDialog* dlg = new PVWidgets::PVMappingPlottingEditDialog(NULL, &plotted.get_plotting(), parent);
	if (dlg->exec() != QDialog::Accepted) {
		return false;
	}

	if (plotted.is_current_plotted()) {
		Picviz::PVPlotted_sp plotted_sp = plotted.shared_from_this();
		PVHive::call<FUNC(Picviz::PVPlotted::process_parent_mapped)>(plotted_sp);
	}

	return true;
}
