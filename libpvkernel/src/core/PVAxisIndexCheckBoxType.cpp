//! \file PVAxisIndexCheckboxType.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

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

