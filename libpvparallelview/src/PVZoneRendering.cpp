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
	_cancel_state = cancel_state::value(false);
}

/******************************************************************************
 *
 * PVParallelView::PVZoneRenderingBase::finished
 *
 *****************************************************************************/
void PVParallelView::PVZoneRenderingBase::finished(p_type const& this_sp)
{
	// Having `this_sp' as parameter allows not to have an internal weak_ptr in PVZoneRenderingBase
	assert(this_sp.get() == this);

	cancel_state state = (cancel_state)_cancel_state;

	// Cancellation state may have been changed in the middle, but the listeners are aware of that!
	// We need to be coherent according to the state at the beggining of this function.
	if (_qobject_finished_success != nullptr && !state.should_cancel()) {
		assert(QThread::currentThread() != _qobject_finished_success->thread());
		const int zone_id = zid();
		QMetaObject::invokeMethod(_qobject_finished_success, _qobject_slot, Qt::QueuedConnection,
				Q_ARG(PVParallelView::PVZoneRenderingBase_p, this_sp),
				Q_ARG(int, zone_id));
	}

	{
		boost::lock_guard<boost::mutex> lock(_wait_mut);
		_finished = true;
	}
	_wait_cond.notify_all();
}
