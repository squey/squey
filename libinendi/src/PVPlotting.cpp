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
		PVPlottingProperties plotting_axis(format, i);
		_columns << plotting_axis;
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
	return _columns[col].get_plotting_filter();
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
	assert(j < _columns.size());
	return get_properties_for_col(j).invalidate();
}

/******************************************************************************
 *
 * Inendi::PVPlotting::is_col_uptodate
 *
 *****************************************************************************/
bool Inendi::PVPlotting::is_col_uptodate(PVCol j) const
{
	assert(j < _columns.size());
	return get_properties_for_col(j).is_uptodate();
}

/******************************************************************************
 *
 * Inendi::PVPlotting::is_uptodate
 *
 *****************************************************************************/
bool Inendi::PVPlotting::is_uptodate() const
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
 * Inendi::PVPlotting::serialize
 *
 *****************************************************************************/
void Inendi::PVPlotting::serialize(PVCore::PVSerializeObject& so,
                                   PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.list("properties", _columns);
	QString name = QString::fromStdString(_name);
	so.attribute("name", name);
}

/******************************************************************************
 *
 * Inendi::PVPlotting::set_uptodate_for_col
 *
 *****************************************************************************/
void Inendi::PVPlotting::set_uptodate_for_col(PVCol j)
{
	assert(j < _columns.size());
	return get_properties_for_col(j).set_uptodate();
}
