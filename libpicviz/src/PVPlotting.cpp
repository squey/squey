//! \file PVPlotting.cpp
//! $Id: PVPlotting.cpp 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvkernel/rush/PVFormat.h>

#include <picviz/PVMapped.h>
#include <picviz/PVPlotting.h>
#include <picviz/PVSource.h>

#include <iostream>

/******************************************************************************
 *
 * Picviz::PVPlotting::PVPlotting
 *
 *****************************************************************************/
Picviz::PVPlotting::PVPlotting(PVMapped* parent)
{
	set_mapped(parent);

	PVRush::PVFormat_p format = parent->get_format();

	for (int i=0; i < format->get_axes().size(); i++) {
		PVPlottingProperties plotting_axis(*format, i);
		_columns << plotting_axis;
		PVLOG_HEAVYDEBUG("%s: Add a column\n", __FUNCTION__);
	}
}

/******************************************************************************
 *
 * Picviz::PVPlotting::~PVPlotting
 *
 *****************************************************************************/
Picviz::PVPlotting::~PVPlotting()
{

}

/******************************************************************************
 *
 * Picviz::PVPlotting::get_format
 *
 *****************************************************************************/
PVRush::PVFormat_p Picviz::PVPlotting::get_format() const
{
	return _mapped->get_format();
}

/******************************************************************************
 *
 * Picviz::PVPlotting::get_qtnraw
 *
 *****************************************************************************/
PVRush::PVNraw::nraw_table& Picviz::PVPlotting::get_qtnraw()
{
	return _mapped->get_qtnraw();
}

const PVRush::PVNraw::nraw_table& Picviz::PVPlotting::get_qtnraw() const
{
	return _mapped->get_qtnraw();
}

/******************************************************************************
 *
 * Picviz::PVPlotting::get_source_parent
 *
 *****************************************************************************/
Picviz::PVSource* Picviz::PVPlotting::get_source_parent()
{
	return _mapped->get_source_parent();
}


Picviz::PVPlottingFilter::p_type Picviz::PVPlotting::get_filter_for_col(PVCol col)
{
	return _columns[col].get_plotting_filter();
}

Picviz::PVMapped* Picviz::PVPlotting::get_mapped_parent()
{
	return _mapped;
}

const Picviz::PVMapped* Picviz::PVPlotting::get_mapped_parent() const
{
	return _mapped;
}

Picviz::PVRoot* Picviz::PVPlotting::get_root_parent()
{
	return _root;
}

void Picviz::PVPlotting::set_mapped(PVMapped* mapped)
{
	_mapped = mapped;
	_root = mapped->get_root_parent();
}

QString const& Picviz::PVPlotting::get_column_type(PVCol col) const
{
	PVMappingProperties const& prop(_mapped->get_mapping().get_properties_for_col(col));
	return prop.get_type();
}

void Picviz::PVPlotting::serialize(PVCore::PVSerializeObject &so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.list("properties", _columns);
}
