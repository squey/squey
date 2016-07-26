/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVFormat.h>

#include <inendi/PVMapped.h>
#include <inendi/PVPlotting.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVSource.h>

// AG: FIXME: see PVLayer.cpp
#include <inendi/PVView.h>

#include <iostream>

/******************************************************************************
 *
 * Inendi::PVPlotting::PVPlotting
 *
 *****************************************************************************/
Inendi::PVPlotting::PVPlotting(PVPlotted* plotted) : _plotted(plotted), _name("default")
{
	PVRush::PVFormat const& format =
	    _plotted->get_parent<Inendi::PVSource>().get_extractor().get_format();

	for (int i = 0; i < format.get_axes().size(); i++) {
		_columns.emplace_back(format, i);
		PVLOG_HEAVYDEBUG("%s: Add a column\n", __FUNCTION__);
	}
}

/******************************************************************************
 *
 * Inendi::PVPlotting::PVPlotting
 *
 *****************************************************************************/
Inendi::PVPlotting::PVPlotting()
{
}

/******************************************************************************
 *
 * Inendi::PVPlotting::~PVPlotting
 *
 *****************************************************************************/
Inendi::PVPlotting::~PVPlotting()
{
}

/******************************************************************************
 *
 * Inendi::PVPlotting::get_filter_for_col
 *
 *****************************************************************************/
Inendi::PVPlottingFilter::p_type Inendi::PVPlotting::get_filter_for_col(PVCol col)
{
	return get_properties_for_col(col).get_plotting_filter();
}

/******************************************************************************
 *
 * Inendi::PVPlotting::get_format
 *
 *****************************************************************************/
PVRush::PVFormat const& Inendi::PVPlotting::get_format() const
{
	return _plotted->get_parent<Inendi::PVSource>().get_extractor().get_format();
}

/******************************************************************************
 *
 * Inendi::PVPlotting::invalidate_column
 *
 *****************************************************************************/
void Inendi::PVPlotting::invalidate_column(PVCol j)
{
	assert((size_t)j < _columns.size());
	return get_properties_for_col(j).invalidate();
}

/******************************************************************************
 *
 * Inendi::PVPlotting::is_col_uptodate
 *
 *****************************************************************************/
bool Inendi::PVPlotting::is_col_uptodate(PVCol j) const
{
	assert((size_t)j < _columns.size());
	return get_properties_for_col(j).is_uptodate();
}

/******************************************************************************
 *
 * Inendi::PVPlotting::is_uptodate
 *
 *****************************************************************************/
bool Inendi::PVPlotting::is_uptodate() const
{
	return std::all_of(_columns.begin(), _columns.end(),
	                   std::mem_fn(&PVPlottingProperties::is_uptodate));
}

/******************************************************************************
 *
 * Inendi::PVPlotting::serialize
 *
 *****************************************************************************/
void Inendi::PVPlotting::serialize(PVCore::PVSerializeObject& so,
                                   PVCore::PVSerializeArchive::version_t /*v*/)
{
	QString name = QString::fromStdString(_name);
	so.attribute("name", name);
	_name = name.toStdString();

	PVCore::PVSerializeObject_p list_obj = so.create_object("properties", "", true, true);

	QString desc_;
	if (so.is_writing()) {
		int idx = 0;
		for (PVPlottingProperties& prop : _columns) {
			QString child_name = QString::number(idx++);
			PVCore::PVSerializeObject_p new_obj = list_obj->create_object(child_name, "", false);
			prop.serialize_write(*new_obj);
			new_obj->_bound_obj = &prop;
			new_obj->_bound_obj_type = typeid(PVPlottingProperties);
		}
	} else {
		int idx = 0;
		try {
			while (true) {
				PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx));
				_columns.emplace_back(PVPlottingProperties::serialize_read(*new_obj));
				new_obj->_bound_obj = &_columns.back();
				new_obj->_bound_obj_type = typeid(PVPlottingProperties);
				idx++;
			}
		} catch (PVCore::PVSerializeArchiveErrorNoObject const& /*e*/) {
		}
	}
}

/******************************************************************************
 *
 * Inendi::PVPlotting::set_uptodate_for_col
 *
 *****************************************************************************/
void Inendi::PVPlotting::set_uptodate_for_col(PVCol j)
{
	assert((size_t)j < _columns.size());
	return get_properties_for_col(j).set_uptodate();
}
