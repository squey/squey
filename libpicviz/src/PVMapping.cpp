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



/******************************************************************************
 *
 * Picviz::PVMapping::PVMapping
 *
 *****************************************************************************/
Picviz::PVMapping::PVMapping(PVSource* parent):
	_name("default")
{
	set_source(parent);

	PVCol naxes = parent->get_column_count();
	if (naxes == 0) {
		PVLOG_ERROR("In PVMapping constructor, no axis have been defined in the format !!!!\n");
		assert(false);
	}

	PVLOG_DEBUG("In PVMapping::PVMapping(), debug PVFormat\n");
	parent->get_rushnraw().format->debug();
	for (PVCol i = 0; i < naxes; i++) {
		PVMappingProperties mapping_axis(*parent->get_rushnraw().format, i);
		columns << mapping_axis;
		PVLOG_HEAVYDEBUG("%s: Add a column\n", __FUNCTION__);
	}
}



/******************************************************************************
 *
 * Picviz::PVMapping::~PVMapping
 *
 *****************************************************************************/
Picviz::PVMapping::~PVMapping()
{
}



/******************************************************************************
 *
 * Picviz::PVMapping::add_column
 *
 *****************************************************************************/
void Picviz::PVMapping::add_column(PVMappingProperties const& props)
{
	columns.push_back(props);
	_mandatory_filters_values.push_back(mandatory_param_map());
}



/******************************************************************************
 *
 * Picviz::PVMapping::clear_trans_nraw
 *
 *****************************************************************************/
void Picviz::PVMapping::clear_trans_nraw()
{
	source->clear_trans_nraw();
}



/******************************************************************************
 *
 * Picviz::PVMapping::get_filter_for_col
 *
 *****************************************************************************/
Picviz::PVMappingFilter::p_type Picviz::PVMapping::get_filter_for_col(PVCol col)
{
	return columns.at(col).get_mapping_filter();
}



/******************************************************************************
 *
 * Picviz::PVMapping::get_format
 *
 *****************************************************************************/
PVRush::PVFormat_p Picviz::PVMapping::get_format() const
{
	return source->get_rushnraw().format;
}



/******************************************************************************
 *
 * Picviz::PVMapping::get_group_key_for_col
 *
 *****************************************************************************/
QString Picviz::PVMapping::get_group_key_for_col(PVCol col) const
{
	return columns[col].get_group_key();
}



/******************************************************************************
 *
 * Picviz::PVMapping::get_mandatory_params_for_col
 *
 *****************************************************************************/
Picviz::mandatory_param_map const& Picviz::PVMapping::get_mandatory_params_for_col(PVCol col) const
{
	assert(col < _mandatory_filters_values.size());
	return _mandatory_filters_values[col];
}



/******************************************************************************
 *
 * Picviz::PVMapping::get_mandatory_params_for_col
 *
 *****************************************************************************/
Picviz::mandatory_param_map& Picviz::PVMapping::get_mandatory_params_for_col(PVCol col)
{
	assert(col < _mandatory_filters_values.size());
	return _mandatory_filters_values[col];
}



/******************************************************************************
 *
 * Picviz::PVMapping::get_mode_for_col
 *
 *****************************************************************************/
QString const& Picviz::PVMapping::get_mode_for_col(PVCol col) const
{
	assert(col < columns.size());
	return get_properties_for_col(col).get_mode();
}



/******************************************************************************
 *
 * Picviz::PVMapping::get_qtnraw
 *
 *****************************************************************************/
PVRush::PVNraw::nraw_table& Picviz::PVMapping::get_qtnraw()
{
	return source->get_qtnraw();
}



/******************************************************************************
 *
 * Picviz::PVMapping::get_qtnraw
 *
 *****************************************************************************/
const PVRush::PVNraw::nraw_table& Picviz::PVMapping::get_qtnraw() const
{
	return source->get_qtnraw();
}



/******************************************************************************
 *
 * Picviz::PVMapping::get_root_parent
 *
 *****************************************************************************/
Picviz::PVRoot* Picviz::PVMapping::get_root_parent() 
{
	return root;
}



/******************************************************************************
 *
 * Picviz::PVMapping::get_root_parent
 *
 *****************************************************************************/
const Picviz::PVRoot* Picviz::PVMapping::get_root_parent() const
{
	return root;
}



/******************************************************************************
 *
 * Picviz::PVMapping::get_source_parent
 *
 *****************************************************************************/
Picviz::PVSource* Picviz::PVMapping::get_source_parent() 
{
	return source;
}



/******************************************************************************
 *
 * Picviz::PVMapping::get_source_parent
 *
 *****************************************************************************/
const Picviz::PVSource* Picviz::PVMapping::get_source_parent() const
{
	return source;
}



/******************************************************************************
 *
 * Picviz::PVMapping::get_trans_nraw
 *
 *****************************************************************************/
PVRush::PVNraw::nraw_trans_table const& Picviz::PVMapping::get_trans_nraw() const
{
	return source->get_trans_nraw();
}



/******************************************************************************
 *
 * Picviz::PVMapping::get_type_for_col
 *
 *****************************************************************************/
QString const& Picviz::PVMapping::get_type_for_col(PVCol col) const
{
	assert(col < columns.size());
	return get_properties_for_col(col).get_type();
}



/******************************************************************************
 *
 * Picviz::PVMapping::invalidate_all
 *
 *****************************************************************************/
void Picviz::PVMapping::invalidate_all()
{
	QList<PVMappingProperties>::iterator it;
	for (it = columns.begin(); it != columns.end(); it++) {
		it->invalidate();
	}
}

/******************************************************************************
 *
 * Picviz::PVMapping::validate_all
 *
 *****************************************************************************/
void Picviz::PVMapping::validate_all()
{
	QList<PVMappingProperties>::iterator it;
	for (it = columns.begin(); it != columns.end(); it++) {
		it->set_uptodate();
	}
}



/******************************************************************************
 *
 * Picviz::PVMapping::is_col_uptodate
 *
 *****************************************************************************/
bool Picviz::PVMapping::is_col_uptodate(PVCol j) const
{
	assert(j < columns.size());
	return get_properties_for_col(j).is_uptodate();
}



/******************************************************************************
 *
 * Picviz::PVMapping::is_uptodate
 *
 *****************************************************************************/
bool Picviz::PVMapping::is_uptodate() const
{
	QList<PVMappingProperties>::const_iterator it;
	for (it = columns.begin(); it != columns.end(); it++) {
		if (!it->is_uptodate()) {
			return false;
		}
	}
	return true;
}



/******************************************************************************
 *
 * Picviz::PVMapping::reset_from_format
 *
 *****************************************************************************/
void Picviz::PVMapping::reset_from_format(PVRush::PVFormat const& format)
{
	PVCol naxes = format.get_axes().size();
	if (columns.size() < naxes) {
		return;
	}

	for (PVCol i = 0; i < naxes; i++) {
		columns[i].set_from_axis(format.get_axes().at(i));
	}
}



/******************************************************************************
 *
 * Picviz::PVMapping::serialize
 *
 *****************************************************************************/
void Picviz::PVMapping::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.list("properties", columns);
	so.attribute("name", _name);
	if (!so.is_writing()) {
		_mandatory_filters_values.clear();
	}
}



/******************************************************************************
 *
 * Picviz::PVMapping::set_default_args
 *
 *****************************************************************************/
void Picviz::PVMapping::set_default_args(PVRush::PVFormat const& format)
{
	QList<PVMappingProperties>::iterator it;
	PVCol i = 0;
	PVCol ncols = format.get_axes().size();
	for (it = columns.begin(); it != columns.end(); it++) {
		it->set_default_args(format.get_axes().at(i));
		i++;
		if (i >= ncols) {
			break;
		}
	}
}



/******************************************************************************
 *
 * Picviz::PVMapping::set_source
 *
 *****************************************************************************/
void Picviz::PVMapping::set_source(PVSource* src)
{
	source = src;
	root = src->get_root();
	
	PVCol naxes = src->get_column_count();
	_mandatory_filters_values.resize(naxes);
}



/******************************************************************************
 *
 * Picviz::PVMapping::set_uptodate_for_col
 *
 *****************************************************************************/
void Picviz::PVMapping::set_uptodate_for_col(PVCol j)
{
	assert(j < columns.size());
	get_properties_for_col(j).set_uptodate();
}



