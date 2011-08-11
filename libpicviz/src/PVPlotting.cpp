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
Picviz::PVPlotting::PVPlotting(PVMapped_p parent)
{
	mapped = parent;
	root = parent->root;

	PVRush::PVFormat_p format = parent->get_format();

	for (int i=0; i < format->get_axes().size(); i++) {
		PVPlottingProperties plotting_axis(root, *format, i);
		columns << plotting_axis;
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
	return mapped->get_format();
}

/******************************************************************************
 *
 * Picviz::PVPlotting::get_qtnraw
 *
 *****************************************************************************/
PVRush::PVNraw::nraw_table& Picviz::PVPlotting::get_qtnraw()
{
	return mapped->get_qtnraw();
}

const PVRush::PVNraw::nraw_table& Picviz::PVPlotting::get_qtnraw() const
{
	return mapped->get_qtnraw();
}

/******************************************************************************
 *
 * Picviz::PVPlotting::get_source_parent
 *
 *****************************************************************************/
Picviz::PVSource_p Picviz::PVPlotting::get_source_parent()
{
	return mapped->get_source_parent();
}


Picviz::PVPlottingFilter::p_type Picviz::PVPlotting::get_filter_for_col(PVCol col)
{
	return columns[col].plotting_filter;
}
