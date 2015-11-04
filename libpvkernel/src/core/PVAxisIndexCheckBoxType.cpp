/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAxisIndexCheckBoxType.h>

PVCore::PVAxisIndexCheckBoxType::PVAxisIndexCheckBoxType()
{
	_origin_axis_index = -1;
}

PVCore::PVAxisIndexCheckBoxType::PVAxisIndexCheckBoxType(int origin_axis_index, bool is_checked)
{
	_origin_axis_index = origin_axis_index;
	_is_checked = is_checked;
}

