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

class PVZoneRenderingBase: public QObject, boost::noncopyable
{
	Q_OBJECT

	typedef std::function<size_t(PVZoneID, PVCore::PVHSVColor const* colors, PVBCICodeBase* codes)> bci_func_type;
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

	PVZoneRenderingBase(bool reversed = false):
		_zid(PVZONEID_INVALID),
		_dst_img(nullptr),
		_width(0),
		_x_start(0),
		_reversed(reversed),
		_finished(false)
	{
		_should_cancel = false;
	}

	virtual ~PVZoneRenderingBase() { }

public:
	inline PVZoneID zid() const { return _zid; }
	inline size_t img_width() const { return _width; }
	inline size_t img_x_start() const { return _x_start; }
	inline float render_zoom_y() const { return _zoom_y; }
	inline bool render_reversed() const { return _reversed; }
	inline bool should_cancel() const { return _should_cancel; }
	inline bool valid() const { return _zid != PVZONEID_INVALID && _width != 0 && _dst_img != nullptr; }
	PVBCIBackendImage& dst_img() const { return *_dst_img; }

	inline void cancel() { _should_cancel = true; }

	inline void set_dst_img(PVBCIBackendImage& dst_img) { assert(_finished); _dst_img = &dst_img; }
	inline void set_zid(PVZoneID const z) { assert(_finished); _zid = z; }
	inline void set_img_width(uint32_t w) { assert(_finished); _width = w; }
	inline void set_img_x_start(uint32_t x) { assert(_finished); _x_start = x; }

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
		_should_cancel = false;
		_finished = false;
	}

signals:
	void render_finished(int zid, bool was_canceled);
	void render_finished_success(int zid);

protected:
	// TODO: implement this
	void finished()
	{
		bool was_canceled;
		{
			boost::lock_guard<boost::mutex> lock(_wait_mut);
			_finished = true;
			was_canceled = _should_cancel;
		}
		_wait_cond.notify_all();

		if (!was_canceled) {
			emit render_finished_success(zid());
		}
		emit render_finished(zid(), was_canceled);
	}

protected:
	inline size_t compute_bci(PVCore::PVHSVColor const* colors, PVBCICodeBase* codes) const { return _f_bci(zid(), colors, codes); }
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
			[=](PVZoneID z, PVCore::PVHSVColor const* colors, PVBCICodeBase* codes)
				{
					return f_bci(z, colors, reinterpret_cast<PVBCICode<Bbits>*>(codes));
				},
			dst_img, x_start, width, zoom_y, reversed)
	{ }
};

}

#endif
