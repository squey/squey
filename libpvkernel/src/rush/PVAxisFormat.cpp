/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <QXmlStreamReader>
#include <QFile>
#include <QHashIterator>
#include <QDateTime>

#include <pvkernel/rush/PVAxisFormat.h>

#include <boost/functional/hash.hpp>

/******************************************************************************
 *
 * PVAxisFormat
 *
 *****************************************************************************/

PVRush::PVAxisFormat::PVAxisFormat(PVCol index) : index(index)
{
}

void PVRush::PVAxisFormat::set_color(QString str)
{
	color.fromQColor(QColor(str));
}

void PVRush::PVAxisFormat::set_color(PVCore::PVColor color_)
{
	color = color_;
}

void PVRush::PVAxisFormat::set_mapping(QString str)
{
	mapping = str;
}

void PVRush::PVAxisFormat::set_name(const QString& str)
{
	name = str;
}

void PVRush::PVAxisFormat::set_plotting(QString str)
{
	plotting = str;
}

void PVRush::PVAxisFormat::set_titlecolor(QString str)
{
	titlecolor.fromQColor(QColor(str));
}

void PVRush::PVAxisFormat::set_titlecolor(PVCore::PVColor color_)
{
	titlecolor = color_;
}

void PVRush::PVAxisFormat::set_type(QString str)
{
	type = str;
}

void PVRush::PVAxisFormat::set_type_format(QString str)
{
	type_format = str;
}
