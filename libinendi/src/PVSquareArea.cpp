/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVSquareArea.h>

Inendi::PVSquareArea::PVSquareArea() : end_x(0), end_y(0), start_x(0), start_y(0), dirty(false)
{
}

/******************************************************************************
 *
 * Inendi::PVSquareArea::get_end_x
 *
 *****************************************************************************/
float Inendi::PVSquareArea::get_end_x() const
{
	return end_x;
}

/******************************************************************************
 *
 * Inendi::PVSquareArea::get_end_y
 *
 *****************************************************************************/
float Inendi::PVSquareArea::get_end_y() const
{
	return end_y;
}

/******************************************************************************
 *
 * Inendi::PVSquareArea::get_start_x
 *
 *****************************************************************************/
float Inendi::PVSquareArea::get_start_x() const
{
	return start_x;
}

/******************************************************************************
 *
 * Inendi::PVSquareArea::get_start_y
 *
 *****************************************************************************/
float Inendi::PVSquareArea::get_start_y() const
{
	return start_y;
}

/******************************************************************************
 *
 * Inendi::PVSquareArea::set_end
 *
 *****************************************************************************/
void Inendi::PVSquareArea::set_end(float ex, float ey)
{
	end_x = ex;
	end_y = ey;
}

/******************************************************************************
 *
 * Inendi::PVSquareArea::set_end_x
 *
 *****************************************************************************/
void Inendi::PVSquareArea::set_end_x(float ex)
{
	end_x = ex;
}

/******************************************************************************
 *
 * Inendi::PVSquareArea::set_end_y
 *
 *****************************************************************************/
void Inendi::PVSquareArea::set_end_y(float ey)
{
	end_y = ey;
}

/******************************************************************************
 *
 * Inendi::PVSquareArea::set_start
 *
 *****************************************************************************/
void Inendi::PVSquareArea::set_start(float sx, float sy)
{
	start_x = sx;
	start_y = sy;
}

/******************************************************************************
 *
 * Inendi::PVSquareArea::set_start_x
 *
 *****************************************************************************/
void Inendi::PVSquareArea::set_start_x(float sx)
{
	start_x = sx;
}

/******************************************************************************
 *
 * Inendi::PVSquareArea::set_start_y
 *
 *****************************************************************************/
void Inendi::PVSquareArea::set_start_y(float sy)
{
	start_y = sy;
}

/******************************************************************************
 *
 * Inendi::PVSquareArea::is_dirty
 *
 *****************************************************************************/
bool Inendi::PVSquareArea::is_dirty() const
{
	return dirty;
}

/******************************************************************************
 *
 * Inendi::PVSquareArea::set_dirty
 *
 *****************************************************************************/
void Inendi::PVSquareArea::set_dirty()
{
	dirty = true;
}

/******************************************************************************
 *
 * Inendi::PVSquareArea::set_clean
 *
 *****************************************************************************/
void Inendi::PVSquareArea::set_clean()
{
	dirty = false;
}

/******************************************************************************
 *
 * Inendi::PVSquareArea::is_empty
 *
 *****************************************************************************/
bool Inendi::PVSquareArea::is_empty() const
{
	return (start_x == end_x) && (start_y == end_y);
}
