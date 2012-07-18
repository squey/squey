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

#include <boost/functional/hash.hpp>


/******************************************************************************
 *
 * PVAxisFormat
 *
 *****************************************************************************/

PVRush::PVAxisFormat::PVAxisFormat()
{
	unique_id_computed = false;
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

void PVRush::PVAxisFormat::set_group(QString str)
{
	group = str;
}

void PVRush::PVAxisFormat::compute_unique_id(QVector<uint32_t> const& tree_ids)
{
	unique_id.clear();
	QVector<uint32_t>::const_iterator it;
	for (it = tree_ids.begin(); it != tree_ids.end(); it++) {
		unique_id.push(*it);
	}
	unique_id_computed = true;
}
