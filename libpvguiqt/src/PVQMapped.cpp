/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVMapped.h>
#include <inendi/widgets/PVMappingPlottingEditDialog.h>
#include <pvguiqt/PVQMapped.h>

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVHive.h>

bool PVGuiQt::PVQMapped::edit_mapped(Inendi::PVMapped& mapped, QWidget* parent)
{
	PVWidgets::PVMappingPlottingEditDialog* dlg = new PVWidgets::PVMappingPlottingEditDialog(mapped.get_mapping(), NULL, parent);
	if (dlg->exec() != QDialog::Accepted) {
		return false;
	}

	if (mapped.is_current_mapped()) {
		Inendi::PVMapped_sp mapped_sp = mapped.shared_from_this();
		PVHive::call<FUNC(Inendi::PVMapped::process_from_parent_source)>(mapped_sp);
	}

	return true;
}
