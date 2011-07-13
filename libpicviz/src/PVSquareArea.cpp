//! \file PVSquareArea.cpp
//! $Id: PVSquareArea.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/PVSquareArea.h>


Picviz::PVSquareArea::PVSquareArea() :
	end_x(0),
	end_y(0),
	start_x(0),
	start_y(0),
	dirty(false)
{
}

/******************************************************************************
 *
 * Picviz::PVSquareArea::get_end_x
 *
 *****************************************************************************/
float Picviz::PVSquareArea::get_end_x() const
{
	return end_x;
}



/******************************************************************************
 *
 * Picviz::PVSquareArea::get_end_y
 *
 *****************************************************************************/
float Picviz::PVSquareArea::get_end_y() const
{
	return end_y;
}



/******************************************************************************
 *
 * Picviz::PVSquareArea::get_start_x
 *
 *****************************************************************************/
float Picviz::PVSquareArea::get_start_x() const
{
	return start_x;
}



/******************************************************************************
 *
 * Picviz::PVSquareArea::get_start_y
 *
 *****************************************************************************/
float Picviz::PVSquareArea::get_start_y() const
{
	return start_y;
}



/******************************************************************************
 *
 * Picviz::PVSquareArea::set_end
 *
 *****************************************************************************/
void Picviz::PVSquareArea::set_end(float ex, float ey)
{
	end_x = ex;
	end_y = ey;
}



/******************************************************************************
 *
 * Picviz::PVSquareArea::set_end_x
 *
 *****************************************************************************/
void Picviz::PVSquareArea::set_end_x(float ex)
{
	end_x = ex;
}



/******************************************************************************
 *
 * Picviz::PVSquareArea::set_end_y
 *
 *****************************************************************************/
void Picviz::PVSquareArea::set_end_y(float ey)
{
	end_y = ey;
}



/******************************************************************************
 *
 * Picviz::PVSquareArea::set_start
 *
 *****************************************************************************/
void Picviz::PVSquareArea::set_start(float sx, float sy)
{
	start_x = sx;
	start_y = sy;
}



/******************************************************************************
 *
 * Picviz::PVSquareArea::set_start_x
 *
 *****************************************************************************/
void Picviz::PVSquareArea::set_start_x(float sx)
{
	start_x = sx;
}

/******************************************************************************
 *
 * Picviz::PVSquareArea::set_start_y
 *
 *****************************************************************************/
void Picviz::PVSquareArea::set_start_y(float sy)
{
	start_y = sy;
}

/******************************************************************************
 *
 * Picviz::PVSquareArea::is_dirty
 *
 *****************************************************************************/
bool Picviz::PVSquareArea::is_dirty() const
{
	return dirty;
}

/******************************************************************************
 *
 * Picviz::PVSquareArea::set_dirty
 *
 *****************************************************************************/
void Picviz::PVSquareArea::set_dirty()
{
	dirty = true;
}

/******************************************************************************
 *
 * Picviz::PVSquareArea::set_clean
 *
 *****************************************************************************/
void Picviz::PVSquareArea::set_clean()
{
	dirty = false;
}

/******************************************************************************
 *
 * Picviz::PVSquareArea::is_empty
 *
 *****************************************************************************/
bool Picviz::PVSquareArea::is_empty() const
{
	return (start_x == end_x) && (start_y == end_y);
}
