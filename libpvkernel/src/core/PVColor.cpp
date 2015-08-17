/**
 * \file PVColor.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <pvkernel/core/PVColor.h>

PVCore::PVColor PVCore::PVColor::fromRgba(unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
	PVColor color;
	color.x = r;
	color.y = g;
	color.z = b;
	color.w	= a;
	return color;
}

/******************************************************************************
 *
 * PVCore::PVColor::toQColor
 *
 *****************************************************************************/
QColor PVCore::PVColor::toQColor() const
{
	return QColor(qRgba(x, y, z, w));
}

/******************************************************************************
 *
 * PVCore::PVColor::fromQColor
 *
 *****************************************************************************/
void PVCore::PVColor::fromQColor(QColor color)
{
	x = color.red();
	y = color.green();
	z = color.blue();
	w = color.alpha();
}
