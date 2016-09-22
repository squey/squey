/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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

PVRush::PVTags::~PVTags() = default;

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
