/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVColor.h>

/******************************************************************************
 *
 * PVCore::PVColor::toQColor
 *
 *****************************************************************************/
QColor PVCore::PVColor::toQColor() const
{
	return QColor(qRgba(r, g, b, a));
}

/******************************************************************************
 *
 * PVCore::PVColor::fromQColor
 *
 *****************************************************************************/
void PVCore::PVColor::fromQColor(QColor const& color)
{
	r = color.red();
	g = color.green();
	b = color.blue();
	a = color.alpha();
}
