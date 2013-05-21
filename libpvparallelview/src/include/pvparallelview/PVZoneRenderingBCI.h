#ifndef PVPARALLELVIEW_PVZONERENDERINGBCI_H
#define PVPARALLELVIEW_PVZONERENDERINGBCI_H

#include <pvparallelview/PVZoneRendering.h>
#include <pvparallelview/PVZoneRenderingBCI_types.h>

namespace PVParallelView {

class PVZoneRenderingBCIBase: public PVZoneRendering
{
	typedef std::function<size_t(PVZoneID, PVCore::PVHSVColor const* colors, PVBCICodeBase* codes)> bci_func_type;

	friend class PVRenderingPipeline;

public:
	typedef PVZoneRenderingBCIBase_p p_type;

public:
	PVZoneRenderingBCIBase(PVZoneID zone_id, bci_func_type const& f_bci, PVBCIBackendImage& dst_img, uint32_t x_start, size_t width, float zoom_y = 1.0f, bool reversed = false):
		PVZoneRendering(zone_id),
		_f_bci(f_bci),
		_dst_img(&dst_img),
		_width(width),
		_x_start(x_start),
		_zoom_y(zoom_y),
		_reversed(reversed)
	{ }

	PVZoneRenderingBCIBase(bool reversed = false):
		PVZoneRendering(),
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

	inline bool valid() const { return PVZoneRendering::valid() && _width != 0 && _dst_img != nullptr; }

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

}

#endif
