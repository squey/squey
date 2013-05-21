#ifndef PVPARALLELVIEW_PVZONERENDERING_H
#define PVPARALLELVIEW_PVZONERENDERING_H

#include <QMetaType>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIBackendImage_types.h>
#include <pvparallelview/PVBCIDrawingBackend.h>
#include <pvparallelview/PVZoneRendering_types.h>

#include <tbb/atomic.h>

#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>
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

class PVZoneRenderingBase: boost::noncopyable
{
	friend class PVRenderingPipeline;

public:
	typedef PVZoneRenderingBase_p p_type;
	typedef std::function<void(PVZoneRenderingBase&)> on_success_function_type;

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
	PVZoneRenderingBase(PVZoneID zone_id):
		_zone_id(zone_id),
		_finished(false)
	{
		init();
	}

	PVZoneRenderingBase():
		_zone_id(PVZONEID_INVALID),
		_finished(false)
	{
		init();
	}

	virtual ~PVZoneRenderingBase()
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

	inline bool cancel()
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

class PVZoneRenderingBCIBase: public PVZoneRenderingBase
{
	typedef std::function<size_t(PVZoneID, PVCore::PVHSVColor const* colors, PVBCICodeBase* codes)> bci_func_type;

	friend class PVRenderingPipeline;

public:
	typedef PVZoneRenderingBCIBase_p p_type;

public:
	PVZoneRenderingBCIBase(PVZoneID zone_id, bci_func_type const& f_bci, PVBCIBackendImage& dst_img, uint32_t x_start, size_t width, float zoom_y = 1.0f, bool reversed = false):
		PVZoneRenderingBase(zone_id),
		_f_bci(f_bci),
		_dst_img(&dst_img),
		_width(width),
		_x_start(x_start),
		_zoom_y(zoom_y),
		_reversed(reversed)
	{ }

	PVZoneRenderingBCIBase(bool reversed = false):
		PVZoneRenderingBase(),
		_dst_img(nullptr),
		_width(0),
		_x_start(0),
		_reversed(reversed)
	{ }

public:
	PVBCIBackendImage& dst_img() const { return *_dst_img; }
	inline size_t img_width() const { return _width; }
	inline size_t img_x_start() const { return _x_start; }

	inline float render_zoom_y() const { return _zoom_y; }
	inline bool render_reversed() const { return _reversed; }

	inline void set_dst_img(PVBCIBackendImage& dst_img) { assert(finished()); _dst_img = &dst_img; }
	inline void set_img_width(uint32_t w) { assert(finished()); _width = w; }
	inline void set_img_x_start(uint32_t x) { assert(finished()); _x_start = x; }

	inline bool valid() const { return PVZoneRenderingBase::valid() && _width != 0 && _dst_img != nullptr; }

protected:
	inline size_t compute_bci(PVCore::PVHSVColor const* colors, PVBCICodeBase* codes) const { return _f_bci(get_zone_id(), colors, codes); }
	inline void render_bci(PVBCIDrawingBackend& backend, PVBCICodeBase* codes, size_t n, std::function<void()> const& render_done = std::function<void()>())
	{
		backend(*_dst_img, img_x_start(), img_width(), codes, n, render_zoom_y(), render_reversed(), render_done);
	}

private:
	// BCI computing function
	// sizeof(std::function<...>) is 32 bytes.. :/
	bci_func_type _f_bci;

	// Dst image parameters
	PVBCIBackendImage* _dst_img;
	size_t _width;
	uint32_t _x_start;
	float _zoom_y;

	bool _reversed;
};

// Helper class
template <size_t Bbits = NBITS_INDEX>
class PVZoneRenderingBCI: public PVZoneRenderingBCIBase
{
public:
	template <class Fbci>
	PVZoneRenderingBCI(PVZoneID zone_id, Fbci const& f_bci, PVBCIBackendImage& dst_img, uint32_t x_start, size_t width, float zoom_y = 1.0f, bool reversed = false):
		PVZoneRenderingBCIBase(zone_id,
			[=](PVZoneID z, PVCore::PVHSVColor const* colors, PVBCICodeBase* codes)
				{
					return f_bci(z, colors, reinterpret_cast<PVBCICode<Bbits>*>(codes));
				},
			dst_img, x_start, width, zoom_y, reversed)
	{ }
};

template <size_t Bbits>
using PVZoneRenderingBCI_p = boost::shared_ptr<PVZoneRenderingBCI<Bbits>>;

}

Q_DECLARE_METATYPE(PVParallelView::PVZoneRenderingBase_p)

#endif
