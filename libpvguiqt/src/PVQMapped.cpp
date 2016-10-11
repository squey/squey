/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVMapped.h>
#include <inendi/widgets/PVMappingPlottingEditDialog.h>
#include <pvguiqt/PVQMapped.h>
#include <pvkernel/core/PVProgressBox.h>

bool PVGuiQt::PVQMapped::edit_mapped(Inendi::PVMapped& mapped, QWidget* parent)
{
	PVWidgets::PVMappingPlottingEditDialog dlg(&mapped, nullptr, parent);
	if (dlg.exec() != QDialog::Accepted) {
		return false;
	}

	PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& /*pbox*/) { mapped.update_mapping(); },
	    QObject::tr("Updating mapping(s)..."), nullptr);

	return true;
}
