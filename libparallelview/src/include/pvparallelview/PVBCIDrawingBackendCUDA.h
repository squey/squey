#ifndef PVPARALLELVIEW_PVBCIDRAWINGBACKENDCUDA_H
#define PVPARALLELVIEW_PVBCIDRAWINGBACKENDCUDA_H

#include <pvparallelview/PVBCIDrawingBackend.h>

namespace PVParallelView {

class PVBCIDrawingBackendCUDA: public PVBCIDrawingBackend
{
public:
	typedef uint32_t pixel_t;
	typedef uint32_t* pixel_pointer_t;

public:
	PVBCIDrawingBackendCUDA();
	virtual ~PVBCIDrawingBackendCUDA();

public:
	PVBCIBackendImage_p create_image(size_t img_width) const;
	void operator()(PVBCIBackendImage& dst_img, size_t x_start, size_t width, PVBCICode* codes, size_t n) const;

private:
	// TODO: make that one per CUDA device !
	PVBCICode* _device_codes;
};

}

#endif
