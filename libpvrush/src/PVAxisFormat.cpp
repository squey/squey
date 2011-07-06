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

#include <pvrush/pv_axis_format.h>




/******************************************************************************
 *
 * PVAxisFormat
 *
 *****************************************************************************/

PVAxisFormat::PVAxisFormat()
{

}


PVAxisFormat::~PVAxisFormat()
{

}



void PVAxisFormat::set_color(QString str)
{
	color = str;
}


void PVAxisFormat::set_mapping(QString str)
{
	mapping = str;
}


void PVAxisFormat::set_name(QString str)
{
	name = str;
}


void PVAxisFormat::set_plotting(QString str)
{
	plotting = str;
}


void PVAxisFormat::set_title_color(QString str)
{
	color = str;
}


void PVAxisFormat::set_type(QString str)
{
	type = str;
}







