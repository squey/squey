/**
 * \file PVZoneRenderingBase.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVZoneRendering.h>
#include <QMetaObject>

/******************************************************************************
 *
 * PVParallelView::PVZoneRenderingBase::init
 *
 *****************************************************************************/
void PVParallelView::PVZoneRenderingBase::init()
{
	_should_cancel = false;
	_qobject_finished_success = nullptr;
}

/******************************************************************************
 *
 * PVParallelView::PVZoneRenderingBase::finished
 *
 *****************************************************************************/
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

	if (_qobject_finished_success != nullptr) {
		const PVZoneID zone_id = get_zone_id();
		QMetaObject::invokeMethod(_qobject_finished_success, _qobject_slot, Qt::QueuedConnection,
			Q_ARG(void*, (void*)this),
			Q_ARG(int, zone_id));
	}
	else {
		_wait_cond.notify_all();
	}
}
