/**
 * \file PVBCIBackendImage.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVBCIBACKENDIMAGE_H
#define PVPARALLELVIEW_PVBCIBACKENDIMAGE_H

#include <pvkernel/core/general.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIBackendImage_types.h>

#include <boost/utility.hpp>

#include <QImage>

namespace PVParallelView {

template <size_t Bbits = NBITS_INDEX>
class PVBCIBackendImage : boost::noncopyable
{
	constexpr static uint32_t image_height = ((uint32_t)1)<<Bbits;
public:
	typedef boost::shared_ptr<PVBCIBackendImage> p_type;

protected:
	PVBCIBackendImage(uint32_t width):
		_width(width)
	{ }

public:
	virtual ~PVBCIBackendImage() { }

public:
	virtual QImage qimage() const = 0;
	inline uint32_t width() const { return _width; }
	constexpr static uint32_t height() { return image_height; }
	virtual bool set_width(uint32_t width) { _width = width; return true; }
	inline size_t size_pixel() const { return _width*height(); }

public:
	// TODO: implement this
	void strech(QImage& /*dst*/, uint32_t /*width*/) { }

private:
	uint32_t _width;
};


}

#endif
