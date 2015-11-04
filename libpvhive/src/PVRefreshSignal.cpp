/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvhive/PVRefreshSignal.h>

void PVHive::__impl::PVRefreshSignal::do_sync_atbd_signal(PVObserverBase* o)
{
	emit about_to_be_deleted_signal(o);
	_atbd_sem.release(1);
}

