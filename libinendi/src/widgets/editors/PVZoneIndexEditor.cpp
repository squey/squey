/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVZoneIndexType.h>

#include <inendi/PVView.h>

#include <inendi/widgets/editors/PVZoneIndexEditor.h>

/******************************************************************************
 *
 * PVCore::PVZoneIndexEditor::PVZoneIndexEditor
 *
 *****************************************************************************/
PVWidgets::PVZoneIndexEditor::PVZoneIndexEditor(Inendi::PVView const& view, QWidget *parent):
	QComboBox(parent),
	_view(view)
{
}

/******************************************************************************
 *
 * PVWidgets::PVZoneIndexEditor::~PVZoneIndexEditor
 *
 *****************************************************************************/
PVWidgets::PVZoneIndexEditor::~PVZoneIndexEditor()
{
}

/******************************************************************************
 *
 * PVWidgets::PVZoneIndexEditor::set_zone_index
 *
 *****************************************************************************/
void PVWidgets::PVZoneIndexEditor::set_zone_index(PVCore::PVZoneIndexType zone_index)
{
	clear();
	addItems(_view.get_zones_names_list());
	setCurrentIndex(zone_index.get_zone_index());
}

PVCore::PVZoneIndexType PVWidgets::PVZoneIndexEditor::get_zone_index() const
{
	int index = currentIndex();
	return PVCore::PVZoneIndexType(index);
}
