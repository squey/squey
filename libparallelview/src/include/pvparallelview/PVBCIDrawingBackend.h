#ifndef PVPARALLELVIEW_PVBCIDRAWINGBACKEND_H
#define PVPARALLELVIEW_PVBCIDRAWINGBACKEND_H

namespace PVParallelView {

class PVBCIDrawingBackend
{
public:
	virtual void init_line_image(QImage& dst_img, size_t width) const = 0;
	virtual void operator()(pixel_pointer_t dst_img, PVBCICode* codes, size_t n) const = 0;

protected:
	template <class PixelType>
	static PixelType* get_image_pointer_for_x_position_helper(QImage& dst_img, size_t x)
	{
	}
};

#endif
