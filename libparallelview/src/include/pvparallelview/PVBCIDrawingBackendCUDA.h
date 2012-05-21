#ifndef PVPARALLELVIEW_PVBCIDRAWINGBACKENDCUDA_H
#define PVPARALLELVIEW_PVBCIDRAWINGBACKENDCUDA_H

#include <pvparallelview/PVBCIDrawingBackend.h>

namespace PVParallelView {

class PVBCIDrawingBackendCUDA: public PVBCIDrawingBackend
{
public:
	// Each backend can have its own pixel format, which will be defined in the resulting QImage !
	typedef uint32_t pixel_t;
	typedef uint32_t* pixel_pointer_t;

public:
	void init_line_image(QImage& dst_img, size_t img_width) const;
	void operator()(pixel_pointer_t dst_img, size_t width, PVBCICode* codes, size_t n) const;

private:
	inline static pixel_pointer_t get_image_pointer_for_x_position(QImage& dst_img, size_t x) const
	{
		return PVBCIDrawingBackend::get_image_pointer_for_x_position_helper<pixel_t>(dst_img, x);
	}
};

}

#endif
