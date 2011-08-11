/*
 * $Id: PVAxisFormat.cpp 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 *
 */

#include <QXmlStreamReader>
#include <QFile>
#include <QHashIterator>
#include <QDateTime>

#include <pvkernel/rush/PVAxisFormat.h>




/******************************************************************************
 *
 * PVAxisFormat
 *
 *****************************************************************************/

PVRush::PVAxisFormat::PVAxisFormat()
{
}


PVRush::PVAxisFormat::~PVAxisFormat()
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


void PVRush::PVAxisFormat::set_name(QString str)
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

void PVRush::PVAxisFormat::set_group(QString str)
{
	group = str;
}

void PVRush::PVAxisFormat::set_key(QString str)
{
	_is_key = str.compare("true", Qt::CaseInsensitive);
}
