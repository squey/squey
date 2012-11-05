/**
 * \file PVZoneRenderingBase.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVZoneRendering.h>
#include <QMetaObject>

#include <QThread>

/******************************************************************************
 *
 * PVParallelView::PVZoneRenderingBase::init
 *
 *****************************************************************************/
void PVParallelView::PVZoneRenderingBase::init()
{
	_qobject_finished_success = nullptr;
	_cancel_state = cancel_state::value(false, false);
}

/******************************************************************************
 *
 * PVParallelView::PVZoneRenderingBase::finished
 *
 *****************************************************************************/
bool PVParallelView::PVZoneRenderingBase::finished()
{
	cancel_state state = (cancel_state)_cancel_state;
	if (state.delete_on_finish()) {
		// We can't notify anyone as we are going to be destructed by the calling TBB thread
		return true;
	}

	// Cancellation state may have been changed in the middle, but the listeners are aware of that!
	// We need to be coherent according to the state at the beggining of this function.
	if (_qobject_finished_success != nullptr && !state.should_cancel()) {
		assert(QThread::currentThread() != _qobject_finished_success->thread());
		const int zone_id = zid();
		QMetaObject::invokeMethod(_qobject_finished_success, _qobject_slot, Qt::QueuedConnection,
				Q_ARG(void*, (void*)this),
				Q_ARG(int, zone_id));
	}

	{
		boost::lock_guard<boost::mutex> lock(_wait_mut);
		_finished = true;
	}
	_wait_cond.notify_all();

	return false;
}
