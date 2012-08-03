/**
 * \file PVBCIDrawingBackend.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVBCIDRAWINGBACKEND_H
#define PVPARALLELVIEW_PVBCIDRAWINGBACKEND_H

#include <pvkernel/core/general.h>
#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIBackendImage_types.h>

#include <QImage>

namespace PVParallelView {

template <size_t Bbits>
class PVBCICode;

template <size_t Bbits = NBITS_INDEX>
class PVBCIDrawingBackend
{
public:
	virtual PVBCIBackendImage_p<Bbits> create_image(size_t img_width) const = 0;
	virtual void operator()(PVBCIBackendImage<Bbits>& dst_img, size_t x_start, size_t width, PVBCICode<Bbits>* codes, size_t n) const = 0;

protected:
	template <class PixelType>
	static inline PixelType* get_image_pointer_for_x_position_helper(PixelType* dst_img, size_t x)
	{
		return dst_img+x*PVBCIBackendImage<Bbits>::height();
	}
};

}

#endif
