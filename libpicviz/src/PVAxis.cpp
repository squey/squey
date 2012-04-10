//! \file PVAxis.cpp
//! $Id: PVAxis.cpp 2526 2011-05-02 12:21:26Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/PVAxis.h>
#include <picviz/PVMappingFilter.h>
#include <picviz/PVPlottingFilter.h>

/******************************************************************************
 *
 * Picviz::PVAxis::PVAxis
 *
 *****************************************************************************/
Picviz::PVAxis::PVAxis()
{
	init();
}

Picviz::PVAxis::PVAxis(PVRush::PVAxisFormat const& axis_format) :
	PVRush::PVAxisFormat(axis_format)
{
	init();
}

void Picviz::PVAxis::init()
{
	is_expandable = true;
	is_expanded = false;
	thickness = 1.0;

	// Create mapping arguments
	
	// Get the mapping filter from the library
	{
		Picviz::PVMappingFilter::p_type lib_filter = LIB_CLASS(Picviz::PVMappingFilter)::get().get_class_by_name(get_type() + "_" + get_mapping());
		if (lib_filter) {
			PVCore::PVArgumentList def_args = lib_filter->get_default_args();
			_args_mapping = args_from_node(get_args_mapping_string(), def_args);
		}
	}

	// Same for the plotting filter
	{
		Picviz::PVPlottingFilter::p_type lib_filter = LIB_CLASS(Picviz::PVPlottingFilter)::get().get_class_by_name(get_type() + "_" + get_plotting());
		if (lib_filter) {
			PVCore::PVArgumentList def_args = lib_filter->get_default_args();
			_args_plotting = args_from_node(get_args_plotting_string(), def_args);
		}
	}
}

/******************************************************************************
 *
 * Picviz::PVAxis::serialize
 *
 *****************************************************************************/
void Picviz::PVAxis::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*version*/)
{
	PVLOG_INFO("%s:%d: doing nothing\n", __FILE__, __LINE__);
	so.attribute("is_expandable", is_expandable);
	so.attribute("is_expanded", is_expanded);
	so.attribute("thickness", thickness);
	so.attribute("name", name);
}

/******************************************************************************
 *
 * Picviz::PVAxis::~PVAxis
 *
 *****************************************************************************/
Picviz::PVAxis::~PVAxis()
{

}

PVCore::PVArgumentList Picviz::PVAxis::args_from_node(node_args_t const& args_str, PVCore::PVArgumentList const& def_args)
{
	PVCore::PVArgumentList ret;
	node_args_t::const_iterator it;
	for (it = args_str.begin(); it != args_str.end(); it++) {
		QString const& key(it.key());
		if (def_args.contains(key)) {
			ret[it.key()] = PVCore::QString_to_PVArgument(it.value(), def_args[key]);
		}
	}
	return ret;
}
