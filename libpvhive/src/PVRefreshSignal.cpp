
#include <pvhive/PVRefreshSignal.h>

void PVHive::__impl::PVRefreshSignal::do_sync_atbd_signal(PVObserverBase* o)
{
	emit about_to_be_deleted_signal(o);
	_atbd_sem.release(1);
}
