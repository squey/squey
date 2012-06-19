//! \file PVPlotting.cpp
//! $Id: PVPlotting.cpp 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvkernel/rush/PVFormat.h>

#include <picviz/PVMapped.h>
#include <picviz/PVPlotting.h>
#include <picviz/PVPlotted.h>
#include <picviz/PVSource.h>

#include <iostream>


/******************************************************************************
 *
 * Picviz::PVPlotting::PVPlotting
 *
 *****************************************************************************/
Picviz::PVPlotting::PVPlotting(PVMapped* parent):
	_name("default")
{
	set_parent(parent);

	PVRush::PVFormat_p format = parent->get_format();

	for (int i=0; i < format->get_axes().size(); i++) {
		PVPlottingProperties plotting_axis(*get_parent<PVMapping>(), *format, i);
		_columns << plotting_axis;
		PVLOG_HEAVYDEBUG("%s: Add a column\n", __FUNCTION__);
	}
}

/******************************************************************************
 *
 * Picviz::PVPlotting::PVPlotting
 *
 *****************************************************************************/
Picviz::PVPlotting::PVPlotting() : data_tree_plotting_t() { }

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
 * Picviz::PVPlotting::add_column
 *
 *****************************************************************************/
void Picviz::PVPlotting::add_column(PVPlottingProperties const& props)
{
	_columns.push_back(props);
}



/******************************************************************************
 *
 * Picviz::PVPlotting::get_column_type
 *
 *****************************************************************************/
QString const& Picviz::PVPlotting::get_column_type(PVCol col) const
{
	PVMappingProperties const& prop(get_parent<PVMapping>()->get_properties_for_col(col));
	return prop.get_type();
}



/******************************************************************************
 *
 * Picviz::PVPlotting::get_filter_for_col
 *
 *****************************************************************************/
Picviz::PVPlottingFilter::p_type Picviz::PVPlotting::get_filter_for_col(PVCol col)
{
	return _columns[col].get_plotting_filter();
}



/******************************************************************************
 *
 * Picviz::PVPlotting::get_format
 *
 *****************************************************************************/
PVRush::PVFormat_p Picviz::PVPlotting::get_format() const
{
	return get_parent<PVMapped>()->get_format();
}

/******************************************************************************
 *
 * Picviz::PVPlotting::get_qtnraw
 *
 *****************************************************************************/
PVRush::PVNraw::nraw_table& Picviz::PVPlotting::get_qtnraw()
{
	return get_parent<PVMapped>()->get_qtnraw();
}

const PVRush::PVNraw::nraw_table& Picviz::PVPlotting::get_qtnraw() const
{
	return get_parent<PVMapped>()->get_qtnraw();
}

/******************************************************************************
 *
 * Picviz::PVPlotting::invalidate_column
 *
 *****************************************************************************/
void Picviz::PVPlotting::invalidate_column(PVCol j)
{
	assert(j < _columns.size());
	return get_properties_for_col(j).invalidate();
}	



/******************************************************************************
 *
 * Picviz::PVPlotting::is_col_uptodate
 *
 *****************************************************************************/
bool Picviz::PVPlotting::is_col_uptodate(PVCol j) const
{
	assert(j < _columns.size());
	return get_properties_for_col(j).is_uptodate();
}



/******************************************************************************
 *
 * Picviz::PVPlotting::is_uptodate
 *
 *****************************************************************************/
bool Picviz::PVPlotting::is_uptodate() const
{
	QList<PVPlottingProperties>::const_iterator it;
	for (it = _columns.begin(); it != _columns.end(); it++) {
		if (!it->is_uptodate()) {
			return false;
		}
	}
	return true;
}



/******************************************************************************
 *
 * Picviz::PVPlotting::reset_from_format
 *
 *****************************************************************************/
void Picviz::PVPlotting::reset_from_format(PVRush::PVFormat const& format)
{
	PVCol naxes = format.get_axes().size();
	if (_columns.size() < naxes) {
		return;
	}

	for (PVCol i = 0; i < naxes; i++) {
		_columns[i].set_from_axis(format.get_axes().at(i));
	}
}



/******************************************************************************
 *
 * Picviz::PVPlotting::serialize
 *
 *****************************************************************************/
void Picviz::PVPlotting::serialize(PVCore::PVSerializeObject &so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.list("properties", _columns);
	so.attribute("name", _name);
}



/******************************************************************************
 *
 * Picviz::PVPlotting::set_mapped
 *
 *****************************************************************************/
void Picviz::PVPlotting::set_parent(PVMapped* mapped)
{
	data_tree_plotting_t::set_parent(mapped);

	// Set parent mapping for properties
	QList<PVPlottingProperties>::iterator it;
	for (it = _columns.begin(); it != _columns.end(); it++) {
		it->set_mapping(*get_parent<PVMapping>());
	}
}



/******************************************************************************
 *
 * Picviz::PVPlotting::set_uptodate_for_col
 *
 *****************************************************************************/
void Picviz::PVPlotting::set_uptodate_for_col(PVCol j)
{
	assert(j < _columns.size());
	return get_properties_for_col(j).set_uptodate();
}	
