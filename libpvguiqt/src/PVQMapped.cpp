#include <picviz/PVMapped.h>
#include <picviz/widgets/PVMappingPlottingEditDialog.h>
#include <pvguiqt/PVQMapped.h>

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVHive.h>

bool PVGuiQt::PVQMapped::edit_mapped(Picviz::PVMapped& mapped, QWidget* parent)
{
	PVWidgets::PVMappingPlottingEditDialog* dlg = new PVWidgets::PVMappingPlottingEditDialog(mapped.get_mapping(), NULL, parent);
	if (dlg->exec() != QDialog::Accepted) {
		return false;
	}

	if (mapped.is_current_mapped()) {
		Picviz::PVMapped_sp mapped_sp = mapped.shared_from_this();
		PVHive::call<FUNC(Picviz::PVMapped::process_from_parent_source)>(mapped_sp);
	}

	return true;
}
