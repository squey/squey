/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVOriginalAxisIndexType.h>

PVCore::PVOriginalAxisIndexType::PVOriginalAxisIndexType()
    : _origin_axis_index(-1), _append_none_axis(false)
{
}

PVCore::PVOriginalAxisIndexType::PVOriginalAxisIndexType(PVCol origin_axis_index,
                                                         bool append_none_axis)
    : _origin_axis_index(origin_axis_index), _append_none_axis(append_none_axis)
{
}

PVCol PVCore::PVOriginalAxisIndexType::get_original_index() const
{
	return _origin_axis_index;
}

bool PVCore::PVOriginalAxisIndexType::get_append_none_axis() const
{
	return _append_none_axis;
}
