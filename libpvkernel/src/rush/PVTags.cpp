/*
 * $Id$
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 *
 */

#include <pvkernel/rush/PVTags.h>

/******************************************************************************
 *
 * PVTags
 *
 *****************************************************************************/
PVRush::PVTags::PVTags()
{

}


PVRush::PVTags::~PVTags()
{

}

void PVRush::PVTags::add_tag(QString tag)
{
	_tags << tag;
}

bool PVRush::PVTags::del_tag(QString tag)
{
	return _tags.remove(tag);
}

bool PVRush::PVTags::has_tag(QString tag) const
{
	return _tags.contains(tag);
}
