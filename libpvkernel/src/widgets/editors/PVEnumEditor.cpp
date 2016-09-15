/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVEnumType.h>              // for PVEnumType
#include <pvkernel/widgets/editors/PVEnumEditor.h> // for PVEnumEditor

/******************************************************************************
 *
 * PVWidgets::PVEnumEditor::set_enum
 *
 *****************************************************************************/
void PVWidgets::PVEnumEditor::set_enum(PVCore::PVEnumType e)
{
	_e = e;
	clear();
	addItems(e.get_list());
	setCurrentIndex(e.get_sel_index());
}

PVCore::PVEnumType PVWidgets::PVEnumEditor::get_enum() const
{
	PVCore::PVEnumType ret(_e);
	ret.set_sel(currentIndex());
	return ret;
}
