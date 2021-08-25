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

#ifndef PVPARALLELVIEW_PVZONERENDERINGBCI_H
#define PVPARALLELVIEW_PVZONERENDERINGBCI_H

#include <pvparallelview/PVZoneRendering.h>
#include <pvparallelview/PVZoneRenderingBCI_types.h>

namespace PVParallelView
{

/**
 * Class providing facilities to compute BCI values and images from this.
 */
class PVZoneRenderingBCIBase : public PVZoneRendering
{
	typedef std::function<size_t(PVZoneID, PVCore::PVHSVColor const* colors, PVBCICodeBase* codes)>
	    bci_func_type;

	friend class PVRenderingPipeline;

  public:
	typedef PVZoneRenderingBCIBase_p p_type;

  public:
	PVZoneRenderingBCIBase(PVZoneID zone_id,
	                       bci_func_type f_bci,
	                       PVBCIBackendImage_p& dst_img,
	                       uint32_t x_start,
	                       size_t width,
	                       float zoom_y = 1.0f,
	                       bool reversed = false)
	    : PVZoneRendering(zone_id)
	    , _f_bci(std::move(f_bci))
	    , _dst_img(dst_img)
	    , _width(width)
	    , _x_start(x_start)
	    , _zoom_y(zoom_y)
	    , _reversed(reversed)
	{
	}

	explicit PVZoneRenderingBCIBase(bool reversed = false)
	    : PVZoneRendering(), _dst_img(nullptr), _width(0), _x_start(0), _reversed(reversed)
	{
	}

  public:
	PVBCIBackendImage& dst_img() const { return *_dst_img; }
	inline size_t img_width() const { return _width; }
	inline size_t img_x_start() const { return _x_start; }

	inline float render_zoom_y() const { return _zoom_y; }
	inline bool render_reversed() const { return _reversed; }

	inline void set_dst_img(PVBCIBackendImage_p& dst_img)
	{
		assert(finished());
		_dst_img = dst_img;
	}

	inline bool valid() const
	{
		return PVZoneRendering::valid() && _width != 0 && _dst_img != nullptr;
	}

  protected:
	inline size_t compute_bci(PVCore::PVHSVColor const* colors, PVBCICodeBase* codes) const
	{
		return _f_bci(get_zone_id(), colors, codes);
	}

	/**
	 * Perform image computation from BCI codes.
	 */
	inline void render_bci(PVBCIDrawingBackend& backend,
	                       PVBCICodeBase* codes,
	                       size_t n,
	                       std::function<void()> const& render_done = std::function<void()>())
	{
		backend.render(_dst_img, img_x_start(), img_width(), codes, n, render_zoom_y(),
		               render_reversed(), render_done);
	}

  private:
	// BCI computing function
	// sizeof(std::function<...>) is 32 bytes.. :/
	bci_func_type _f_bci;

	// Dst image parameters
	PVBCIBackendImage_p _dst_img;
	size_t _width;
	uint32_t _x_start;
	float _zoom_y;

	bool _reversed;
};

// Helper class
template <size_t Bbits = NBITS_INDEX>
class PVZoneRenderingBCI : public PVZoneRenderingBCIBase
{
  public:
	template <class Fbci>
	PVZoneRenderingBCI(PVZoneID zone_id,
	                   Fbci const& f_bci,
	                   PVBCIBackendImage_p& dst_img,
	                   uint32_t x_start,
	                   size_t width,
	                   float zoom_y = 1.0f,
	                   bool reversed = false)
	    : PVZoneRenderingBCIBase(
	          zone_id,
	          [=](PVZoneID z, PVCore::PVHSVColor const* colors, PVBCICodeBase* codes) {
		          return f_bci(z, colors, reinterpret_cast<PVBCICode<Bbits>*>(codes));
	          },
	          dst_img,
	          x_start,
	          width,
	          zoom_y,
	          reversed)
	{
	}
};
} // namespace PVParallelView

#endif
