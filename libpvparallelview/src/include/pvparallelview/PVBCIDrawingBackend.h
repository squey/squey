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

#include <functional>

namespace PVParallelView {

class PVBCICodeBase;

class PVBCIDrawingBackend
{
public:
	typedef PVBCIBackendImage backend_image_t;
	typedef PVBCIBackendImage_p backend_image_p_t;

	typedef enum {
		Serial = 1,
		Parallel = 2
	} Flags;

public:
	virtual ~PVBCIDrawingBackend() { }

public:
	virtual backend_image_p_t create_image(size_t img_width, uint8_t height_bits) const = 0;
	virtual Flags flags() const = 0;
	virtual bool is_sync() const = 0;

public:
	// If this backend is synchronous, render_done must be ignored.
	virtual void operator()(PVBCIBackendImage& dst_img, size_t x_start, size_t width, PVBCICodeBase* codes, size_t n, const float zoom_y = 1.0f, bool reverse = false, std::function<void()> const& render_done = std::function<void()>()) = 0;

protected:
/*
	template <class PixelType>
	static inline PixelType* get_image_pointer_for_x_position_helper(PixelType* dst_img, size_t x)
	{
		return dst_img+x*dst_img->height();
	}
	*/
};

class PVBCIDrawingBackendSync: public PVBCIDrawingBackend
{
public:
	bool is_sync() const override { return true; }
};

class PVBCIDrawingBackendAsync: public PVBCIDrawingBackend
{
public:
	bool is_sync() const override { return false; }

public:
	virtual void wait_all() = 0;
};

}

#endif
