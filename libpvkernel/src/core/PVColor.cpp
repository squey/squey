//! \file PVColor.cpp
//! $Id: PVColor.cpp 2492 2011-04-25 05:41:54Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QRgb>

#include <pvkernel/core/PVColor.h>



/******************************************************************************
 *
 * PVCore::PVColor::PVColor
 *
 *****************************************************************************/
PVCore::PVColor::PVColor()
{

}

/******************************************************************************
 *
 * PVCore::PVColor::PVColor
 *
 *****************************************************************************/
PVCore::PVColor::PVColor(unsigned char r, unsigned char g, unsigned char b)
{
	x = r;
	y = g;
	z = b;
}

/******************************************************************************
 *
 * PVCore::PVColor::PVColor
 *
 *****************************************************************************/
PVCore::PVColor::PVColor(unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
	x = r;
	y = g;
	z = b;
	w = a;
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

/******************************************************************************
 *
 * PVCore::PVColor::r
 *
 *****************************************************************************/
unsigned char &PVCore::PVColor::r()
{
	return x;
}

/******************************************************************************
 *
 * PVCore::PVColor::g
 *
 *****************************************************************************/
unsigned char &PVCore::PVColor::g()
{
	return y;
}

/******************************************************************************
 *
 * PVCore::PVColor::b
 *
 *****************************************************************************/
unsigned char &PVCore::PVColor::b()
{
	return z;
}

/******************************************************************************
 *
 * PVCore::PVColor::a
 *
 *****************************************************************************/
unsigned char &PVCore::PVColor::a()
{
	return w;
}

