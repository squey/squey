//! \file PVMapping.cpp
//! $Id: PVMapping.cpp 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvkernel/core/debug.h>

#include <pvkernel/rush/PVFormat.h>
#include <picviz/PVMapping.h>
#include <picviz/PVSource.h>

#include <iostream>

Picviz::PVMapping::PVMapping(PVSource* parent)
{
	set_source(parent);

	PVCol naxes = parent->get_rushnraw().format->get_axes().size();
	if (naxes == 0) {
		PVLOG_ERROR("In PVMapping constructor, no axis have been defined in the format !!!!\n");
		assert(false);
	}

	PVLOG_DEBUG("In PVMapping::PVMapping(), debug PVFormat\n");
	parent->get_rushnraw().format->debug();
	for (int i=0; i < naxes; i++) {
		PVMappingProperties mapping_axis(*parent->get_rushnraw().format, i);
		columns << mapping_axis;
		PVLOG_HEAVYDEBUG("%s: Add a column\n", __FUNCTION__);
	}
}

Picviz::PVMapping::~PVMapping()
{
}

PVRush::PVFormat_p Picviz::PVMapping::get_format() const
{
	return source->get_rushnraw().format;
}

PVRush::PVNraw::nraw_table& Picviz::PVMapping::get_qtnraw()
{
	return source->get_qtnraw();
}

const PVRush::PVNraw::nraw_table& Picviz::PVMapping::get_qtnraw() const
{
	return source->get_qtnraw();
}

Picviz::PVRoot* Picviz::PVMapping::get_root_parent() 
{
	return root;
}

Picviz::PVSource* Picviz::PVMapping::get_source_parent() 
{
	return source;
}

const Picviz::PVRoot* Picviz::PVMapping::get_root_parent() const
{
	return root;
}

const Picviz::PVSource* Picviz::PVMapping::get_source_parent() const
{
	return source;
}

PVRush::PVNraw::nraw_trans_table const& Picviz::PVMapping::get_trans_nraw() const
{
	return source->get_trans_nraw();
}

void Picviz::PVMapping::clear_trans_nraw()
{
	source->clear_trans_nraw();
}

Picviz::PVMappingFilter::p_type Picviz::PVMapping::get_filter_for_col(PVCol col)
{
	return columns.at(col).get_mapping_filter();
}

Picviz::mandatory_param_map const& Picviz::PVMapping::get_mandatory_params_for_col(PVCol col) const
{
	assert(col < _mandatory_filters_values.size());
	return _mandatory_filters_values[col];
}

Picviz::mandatory_param_map& Picviz::PVMapping::get_mandatory_params_for_col(PVCol col)
{
	assert(col < _mandatory_filters_values.size());
	return _mandatory_filters_values[col];
}

QString Picviz::PVMapping::get_group_key_for_col(PVCol col) const
{
	return columns[col].get_group_key();
}

void Picviz::PVMapping::set_source(PVSource* src)
{
	source = src;
	root = src->get_root();
	
	PVCol naxes = src->get_rushnraw().format->get_axes().size();
	_mandatory_filters_values.resize(naxes);
}

void Picviz::PVMapping::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.list("properties", columns);
	if (!so.is_writing()) {
		_mandatory_filters_values.clear();
	}
}
