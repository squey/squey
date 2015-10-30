/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvhive/PVActorBase.h>
#include <pvhive/PVHive.h>

PVHive::PVActorBase::~PVActorBase()
{
	if (_object != nullptr) {
		PVHive::get().unregister_actor(*this);
	}
}
