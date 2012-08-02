/**
 * \file PVAxis.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

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
