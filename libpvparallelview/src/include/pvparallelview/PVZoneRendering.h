#ifndef PVPARALLELVIEW_PVZONERENDERING_H
#define PVPARALLELVIEW_PVZONERENDERING_H

#include <QMetaType>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIBackendImage_types.h>
#include <pvparallelview/PVBCIDrawingBackend.h>
#include <pvparallelview/PVZoneRendering_types.h>

#include <tbb/atomic.h>

#include <boost/utility.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

#include <tbb/spin_rw_mutex.h>

#include <functional>

namespace PVCore {
class PVHSVColor;
}

namespace PVParallelView {

class PVRenderingPipeline;
class PVZonesProcessor;

template <size_t Bbits>
class PVBCICode;

class PVZoneRendering: boost::noncopyable
{
	friend class PVRenderingPipeline;

public:
	typedef PVZoneRendering_p p_type;
	typedef std::function<void(PVZoneRendering&)> on_success_function_type;

private:
	struct cancel_state
	{   
		union {
			struct {
				uint8_t should_cancel: 1;
			} s;
			uint8_t v;
		};   

		bool should_cancel() const { return s.should_cancel; }
		//bool delete_on_finish() const { return s.delete_on_finish; }

		static cancel_state value(bool cancel) { cancel_state ret; ret.v = cancel; return ret; } 
		//static cancel_state value(bool cancel, bool del) { cancel_state ret; ret.v = (cancel | (del<<1)); return ret; }
	};

	struct next_job
	{
		next_job()
		{
			zp = nullptr;
		}

		void launch();

		tbb::atomic<PVZonesProcessor*> zp;
		p_type zr;
	};

public:
	PVZoneRendering(PVZoneID zone_id):
		_zone_id(zone_id),
		_finished(false)
	{
		init();
	}

	PVZoneRendering():
		_zone_id(PVZONEID_INVALID),
		_finished(false)
	{
		init();
	}

	virtual ~PVZoneRendering()
	{
		assert(_job_after_canceled.zp == nullptr);
	}

public:
	inline PVZoneID get_zone_id() const { return _zone_id; }
	inline PVZoneID zid() const { return _zone_id; }
	inline void set_zone_id(PVZoneID const zone_id) { assert(_finished); _zone_id = zone_id; }

	inline bool should_cancel() const
	{
		return ((cancel_state)_cancel_state).should_cancel();
	}
	inline bool valid() const { return _zone_id != (PVZoneID) PVZONEID_INVALID; }

	virtual bool cancel()
	{
		// Returns true if it was previously canceled
		return _cancel_state.fetch_and_store(cancel_state::value(true)).should_cancel();
	}

	void cancel_and_add_job(PVZonesProcessor& zp, p_type const& zr);

	inline void set_render_finished_slot(QObject* receiver, const char* slot) { _qobject_finished_success = receiver; _qobject_slot = slot; }

public:
	void wait_end()
	{
		boost::unique_lock<boost::mutex> lock(_wait_mut);
		while (!_finished) {
			_wait_cond.wait(lock);
		}
	}

	void reset()
	{
		_finished = false;
		_cancel_state = cancel_state::value(false);
		assert(_job_after_canceled.zp == nullptr);
	}

	bool finished() const;

protected:
	void finished(p_type const& this_sp);

private:
	void init();

private:
	PVZoneID _zone_id;
	
	tbb::atomic<cancel_state> _cancel_state;

	// Qt signalisation
	QObject* _qobject_finished_success;
	const char* _qobject_slot;

	// Synchronisation
	boost::condition_variable _wait_cond;
	mutable boost::mutex _wait_mut;
	bool _finished;

	// Next job when this one has been canceled
	next_job _job_after_canceled;
};


}

Q_DECLARE_METATYPE(PVParallelView::PVZoneRendering_p)

#endif
