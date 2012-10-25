#ifndef PVPARALLELVIEW_PVZONERENDERING_H
#define PVPARALLELVIEW_PVZONERENDERING_H

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIBackendImage_types.h>
#include <pvparallelview/PVBCIDrawingBackend.h>

#include <tbb/atomic.h>

#include <boost/utility.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

#include <functional>

namespace PVCore {
class PVHSVColor;
}

namespace PVParallelView {

class PVRenderingPipeline;

template <size_t Bbits>
class PVBCICode;

class PVZoneRenderingBase: boost::noncopyable
{
	typedef std::function<size_t(PVCore::PVHSVColor* colors, PVBCICodeBase* codes)> bci_func_type;
	friend class PVRenderingPipeline;

public:
	PVZoneRenderingBase(PVZoneID zid, bci_func_type const& f_bci, PVBCIBackendImage& dst_img, uint32_t x_start, size_t width, float zoom_y = 1.0f, bool reversed = false):
		_zid(zid),
		_f_bci(f_bci),
		_dst_img(&dst_img),
		_width(width),
		_x_start(x_start),
		_zoom_y(zoom_y),
		_reversed(reversed),
		_finished(false)
	{
		_should_cancel = false;
	}

public:
	inline PVZoneID zid() const { return _zid; }
	inline size_t img_width() const { return _width; }
	inline size_t img_x_start() const { return _x_start; }
	inline float render_zoom_y() const { return _zoom_y; }
	inline bool render_reversed() const { return _reversed; }
	inline bool should_cancel() const { return _should_cancel; }
	inline void cancel() { _should_cancel = true; }

public:
	bool wait_end()
	{
		boost::unique_lock<boost::mutex> lock(_wait_mut);
		while (!_finished) {
			_wait_cond.wait(lock);
		}
		// Be ready if we get back in the game again!
		const bool ret = _should_cancel;
		_should_cancel = false;
		_finished = false;
		return !ret;
	}

protected:
	// TODO: implement this
	void finished()
	{
		{
			boost::lock_guard<boost::mutex> lock(_wait_mut);
			_finished = true;
		}
		_wait_cond.notify_all();
	}

protected:
	inline size_t compute_bci(PVCore::PVHSVColor* colors, PVBCICodeBase* codes) const { return _f_bci(colors, codes); }
	inline void render_bci(PVBCIDrawingBackend& backend, PVBCICodeBase* codes, size_t n, std::function<void()> const& render_done = std::function<void()>())
	{
		backend(*_dst_img, img_x_start(), img_width(), codes, n, render_zoom_y(), render_reversed(), render_done);
	}

private:
	PVZoneID _zid;
	
	// BCI computing function
	// sizeof(std::function<...>) is 32 bytes.. :/
	bci_func_type _f_bci;

	// Dst image parameters
	PVBCIBackendImage* _dst_img;
	size_t _width;
	uint32_t _x_start;
	float _zoom_y;

	bool _reversed;
	tbb::atomic<bool> _should_cancel;

	// Synchronisation
	boost::condition_variable _wait_cond;
	boost::mutex _wait_mut;
	bool _finished;
};


// Helper class
template <size_t Bbits = NBITS_INDEX>
class PVZoneRendering: public PVZoneRenderingBase
{
public:
	template <class Fbci>
	PVZoneRendering(PVZoneID zid, Fbci const& f_bci, PVBCIBackendImage& dst_img, uint32_t x_start, size_t width, float zoom_y = 1.0f, bool reversed = false):
		PVZoneRenderingBase(zid,
			[=](PVCore::PVHSVColor* colors, PVBCICodeBase* codes)
				{
					return f_bci(colors, reinterpret_cast<PVBCICode<Bbits>*>(codes));
				},
			dst_img, x_start, width, zoom_y, reversed)
	{ }
};

}

#endif
