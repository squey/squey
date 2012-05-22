#ifndef PVPARALLELVIEW_PVBCIDRAWINGBACKEND_H
#define PVPARALLELVIEW_PVBCIDRAWINGBACKEND_H

#include <pvkernel/core/general.h>
#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIBackendImage_types.h>

#include <QImage>

namespace PVParallelView {

class PVBCICode;

class PVBCIDrawingBackend
{
public:
	virtual PVBCIBackendImage_p create_image(size_t img_width) const = 0;
	virtual void operator()(PVBCIBackendImage& dst_img, size_t x_start, size_t width, PVBCICode* codes, size_t n) const = 0;

protected:
	template <class PixelType>
	static inline PixelType* get_image_pointer_for_x_position_helper(PixelType* dst_img, size_t x)
	{
		return dst_img+x*PARALLELVIEW_IMAGE_HEIGHT;
	}
};

}

#endif
