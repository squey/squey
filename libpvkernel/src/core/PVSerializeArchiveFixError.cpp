/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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
