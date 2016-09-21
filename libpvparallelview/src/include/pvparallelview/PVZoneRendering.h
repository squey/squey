/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVZONERENDERING_H
#define PVPARALLELVIEW_PVZONERENDERING_H

#include <QMetaType>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIBackendImage_types.h>
#include <pvparallelview/PVBCIDrawingBackend.h>
#include <pvparallelview/PVZoneRendering_types.h>

#include <tbb/atomic.h>

#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

#include <tbb/spin_rw_mutex.h>

#include <functional>
#include <atomic>

namespace PVCore
{
class PVHSVColor;
} // namespace PVCore

namespace PVParallelView
{

class PVRenderingPipeline;
class PVZonesProcessor;

template <size_t Bbits>
struct PVBCICode;

/**
 * It looks like this class is a job scheduler for multiple zone rendering on the same ZoneId
 * It call a QtSlot at the end of the job.
 */
class PVZoneRendering
{
	friend class PVRenderingPipeline;

  public:
	typedef PVZoneRendering_p p_type;

  private:
	using cancel_state = bool;

	struct next_job {
		next_job() { zp = nullptr; }

		void launch();

		tbb::atomic<PVZonesProcessor*> zp;
		p_type zr;
	};

  public:
	explicit PVZoneRendering(PVZoneID zone_id)
	    : _zone_id(zone_id)
	    , _should_cancel(false)
	    , _qobject_finished_success(nullptr)
	    , _finished(false)
	{
	}

	PVZoneRendering(PVZoneRendering const&) = delete;
	PVZoneRendering(PVZoneRendering&&) = delete;
	PVZoneRendering& operator=(PVZoneRendering const&) = delete;
	PVZoneRendering& operator=(PVZoneRendering&&) = delete;

	PVZoneRendering() : PVZoneRendering(PVZONEID_INVALID) {}

	virtual ~PVZoneRendering() { assert(_job_after_canceled.zp == nullptr); }

  public:
	inline PVZoneID get_zone_id() const { return _zone_id; }
	inline void set_zone_id(PVZoneID const zone_id)
	{
		assert(_finished);
		_zone_id = zone_id;
	}

	virtual bool cancel() { return _should_cancel.fetch_and_store(true); }
	inline bool should_cancel() const { return _should_cancel; }
	void cancel_and_add_job(PVZonesProcessor& zp, p_type const& zr);

	inline bool valid() const { return _zone_id != (PVZoneID)PVZONEID_INVALID; }

	inline void set_render_finished_slot(QObject* receiver, const char* slot)
	{
		_qobject_finished_success = receiver;
		_qobject_slot = slot;
	}

  public:
	void wait_end()
	{
		boost::unique_lock<boost::mutex> lock(_wait_mut);
		while (!_finished) {
			_wait_cond.wait(lock);
		}
	}

	bool finished() const { return _finished; }

  protected:
	void finished(p_type const& this_sp);

  private:
	PVZoneID _zone_id;

	tbb::atomic<cancel_state> _should_cancel;

	// Qt signalisation
	QObject* _qobject_finished_success;
	const char* _qobject_slot;

	// Synchronisation
	boost::mutex _wait_mut;
	boost::condition_variable _wait_cond;
	bool _finished;

	// Next job when this one has been canceled
	next_job _job_after_canceled;
};
} // namespace PVParallelView

Q_DECLARE_METATYPE(PVParallelView::PVZoneRendering_p)

#endif
