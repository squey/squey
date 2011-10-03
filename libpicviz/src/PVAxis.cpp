//! \file PVAxis.cpp
//! $Id: PVAxis.cpp 2526 2011-05-02 12:21:26Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/PVAxis.h>


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

}

/******************************************************************************
 *
 * Picviz::PVAxis::~PVAxis
 *
 *****************************************************************************/
Picviz::PVAxis::~PVAxis()
{

}
