/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVAxis.h>
#include <inendi/PVMappingFilter.h>
#include <inendi/PVPlottingFilter.h>

/******************************************************************************
 *
 * Inendi::PVAxis::PVAxis
 *
 *****************************************************************************/
Inendi::PVAxis::PVAxis(PVRush::PVAxisFormat const& axis_format) : PVRush::PVAxisFormat(axis_format)
{
	// Create mapping arguments

	// Get the mapping filter from the library
	{
		Inendi::PVMappingFilter::p_type lib_filter =
		    LIB_CLASS(Inendi::PVMappingFilter)::get().get_class_by_name(get_mapping());
		auto usable_type = lib_filter->list_usable_type();
		if (usable_type.find(get_type().toStdString()) == usable_type.end()) {
			throw std::runtime_error("You can't use this mapping with this type");
		}
		PVCore::PVArgumentList def_args = lib_filter->get_default_args();
		_args_mapping = args_from_node(get_args_mapping_string(), def_args);
	}

	// Same for the plotting filter
	{
		Inendi::PVPlottingFilter::p_type lib_filter =
		    LIB_CLASS(Inendi::PVPlottingFilter)::get().get_class_by_name(get_plotting());
		auto usable_type = lib_filter->list_usable_type();
		if (std::find(usable_type.begin(), usable_type.end(),
		              std::make_pair(get_type().toStdString(), get_mapping().toStdString())) ==
		    usable_type.end()) {
			throw std::runtime_error("You can't use this type/mapping with this plotting");
		}
		PVCore::PVArgumentList def_args = lib_filter->get_default_args();
		_args_plotting = args_from_node(get_args_plotting_string(), def_args);
	}
}

/******************************************************************************
 *
 * Inendi::PVAxis::serialize
 *
 *****************************************************************************/
void Inendi::PVAxis::serialize(PVCore::PVSerializeObject& so,
                               PVCore::PVSerializeArchive::version_t /*version*/)
{
	so.attribute("name", name);
	// so.attribute("color", color);
	// so.attribute("titlecolor", titlecolor);
}

/******************************************************************************
 *
 * Inendi::PVAxis::~PVAxis
 *
 *****************************************************************************/
Inendi::PVAxis::~PVAxis()
{
}

PVCore::PVArgumentList Inendi::PVAxis::args_from_node(node_args_t const& args_str,
                                                      PVCore::PVArgumentList const& def_args)
{
	PVCore::PVArgumentList ret;
	node_args_t::const_iterator it;
	for (it = args_str.begin(); it != args_str.end(); it++) {
		QString const& key(it.key());
		if (def_args.contains(key)) {
			ret[it.key()] = PVCore::QString_to_PVArgument(it.value(), def_args.at(key));
		}
	}
	return ret;
}
