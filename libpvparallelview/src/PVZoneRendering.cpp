/**
 * \file PVZoneRenderingBase.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVZonesProcessor.h>
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

	{
		boost::lock_guard<boost::mutex> lock(_wait_mut);
		_finished = true;
	}

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

	_wait_cond.notify_all();

	// Reread the cancellation state here, as it may have changed in the middle
	// but we still need to launch the job.
	// The "launch()" method will atomically check that the job is launched
	// only once. 
	if (should_cancel()) {
		_job_after_canceled.launch();
	}
}

bool PVParallelView::PVZoneRenderingBase::finished() const
{
	bool ret;
	{
		boost::lock_guard<boost::mutex> lock(_wait_mut);
		ret = _finished;
	}
	return ret;
}

void PVParallelView::PVZoneRenderingBase::cancel_and_add_job(PVZonesProcessor& zp, p_type const& zr)
{
	_job_after_canceled.zr = zr;
	_job_after_canceled.zp = &zp;
	if (cancel() || finished()) {
		// We are already canceled or finished, so just launch the job.
		_job_after_canceled.launch();
	}
}

void PVParallelView::PVZoneRenderingBase::next_job::launch()
{
	PVZonesProcessor* const zp_ = zp.fetch_and_store(nullptr);
	if (zp_) {
		zp_->add_job(zr);
		zr.reset();
	}
}
