#include <pvparallelview/PVZoneRendering.h>
#include <QMetaObject>

void PVParallelView::PVZoneRenderingBase::init()
{
	_should_cancel = false;
	// Move this object to the main thread instance !
	//moveToThread(Qapplication::instance()->thread());
}

void PVParallelView::PVZoneRenderingBase::finished()
{
	bool was_canceled;
	{
		boost::lock_guard<boost::mutex> lock(_wait_mut);
		_finished = true;
		was_canceled = _should_cancel;
	}

	if (was_canceled) {
		return;
	}

	if (_qobject_finished_success) {
		const int zone_id = zid();
		QMetaObject::invokeMethod(_qobject_finished_success, _qobject_slot, Qt::QueuedConnection,
			Q_ARG(void*, (void*)this),
			Q_ARG(int, zone_id));
	}
	else {
		_wait_cond.notify_all();
	}
}
