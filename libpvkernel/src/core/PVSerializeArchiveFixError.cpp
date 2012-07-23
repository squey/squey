/**
 * \file PVSerializeArchiveFixError.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVSerializeArchiveFixError.h>
#include <pvkernel/core/PVSerializeObject.h>

void PVCore::PVSerializeArchiveFixError::error_fixed()
{
	_so.error_fixed(this);
}

void PVCore::PVSerializeArchiveFixAttribute::fix(QVariant const& v)
{
	_so.fix_attribute(_attribute, v);
	error_fixed();
}
