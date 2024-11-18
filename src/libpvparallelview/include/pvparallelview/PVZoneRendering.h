/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVPARALLELVIEW_PVZONERENDERING_H
#define PVPARALLELVIEW_PVZONERENDERING_H

#include <QMetaType>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIBackendImage_types.h>
#include <pvparallelview/PVBCIDrawingBackend.h>
#include <pvparallelview/PVZoneRendering_types.h>

#include <atomic>

#include <mutex>
#include <condition_variable>

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

		std::atomic<PVZonesProcessor*> zp;
		p_type zr;
	};

  public:
	explicit PVZoneRendering(PVZoneID zone_id = PVZONEID_INVALID)
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

	virtual ~PVZoneRendering()
	{
		assert(_job_after_canceled.zp == nullptr);
		set_render_finished_slot(nullptr, nullptr);
		cancel();
		wait_end();
	}

  public:
	inline PVZoneID get_zone_id() const { return _zone_id; }
	inline void set_zone_id(PVZoneID const zone_id)
	{
		assert(_finished);
		_zone_id = zone_id;
	}

	virtual bool cancel() { return _should_cancel.exchange(true); }
	inline bool should_cancel() const { return _should_cancel; }
	void cancel_and_add_job(PVZonesProcessor& zp, p_type const& zr);

	inline bool valid() const { return _zone_id != PVZONEID_INVALID; }

	inline void set_render_finished_slot(QObject* receiver, const char* slot)
	{
		std::unique_lock<std::mutex> lock(_wait_mut);
		_qobject_finished_success = receiver;
		_qobject_slot = slot;
	}

  public:
	void wait_end()
	{
		std::unique_lock<std::mutex> lock(_wait_mut);
		while (!_finished) {
			_wait_cond.wait(lock);
		}
	}

	bool finished() const { return _finished; }

  protected:
	void finished(p_type const& this_sp);

  private:
	PVZoneID _zone_id;

	std::atomic<cancel_state> _should_cancel;

	// Qt signalisation
	QObject* _qobject_finished_success;
	const char* _qobject_slot;

	// Synchronisation
	std::mutex _wait_mut;
	std::condition_variable _wait_cond;
	bool _finished;

	// Next job when this one has been canceled
	next_job _job_after_canceled;
};
} // namespace PVParallelView

Q_DECLARE_METATYPE(PVParallelView::PVZoneRendering_p)

#endif
