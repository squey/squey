/**
 * \file PVZoneIndexEditor.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVZoneIndexType.h>

#include <picviz/PVView.h>

#include <picviz/widgets/editors/PVZoneIndexEditor.h>

/******************************************************************************
 *
 * PVCore::PVZoneIndexEditor::PVZoneIndexEditor
 *
 *****************************************************************************/
PVWidgets::PVZoneIndexEditor::PVZoneIndexEditor(Picviz::PVView const& view, QWidget *parent):
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
