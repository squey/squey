/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVBCIDRAWINGBACKEND_H
#define PVPARALLELVIEW_PVBCIDRAWINGBACKEND_H

#include <pvkernel/core/general.h>
#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIBackendImage_types.h>

#include <functional>

namespace PVParallelView
{

class PVBCICodeBase;

/**
 * Management class to create image representation and fill it.
 */
class PVBCIDrawingBackend
{
  public:
	using backend_image_t = PVBCIBackendImage;
	using backend_image_p_t = PVBCIBackendImage_p;

	typedef enum { Serial = 1, Parallel = 2 } Flags;

  public:
	virtual ~PVBCIDrawingBackend() {}

  public:
	virtual backend_image_p_t create_image(size_t img_width, uint8_t height_bits) = 0;
	// TODO : flags is only Serial.
	virtual Flags flags() const = 0;
	virtual bool is_sync() const = 0;

  public:
	virtual PVBCICodeBase* allocate_bci(size_t n)
	{
		return (PVBCICodeBase*)PVBCICode<>::allocate_codes(n);
	}
	virtual void free_bci(PVBCICodeBase* buf) { return PVBCICode<>::free_codes((PVBCICode<>*)buf); }

  public:
	// If this backend is synchronous, render_done must be ignored.
	virtual void operator()(PVBCIBackendImage_p& dst_img,
	                        size_t x_start,
	                        size_t width,
	                        PVBCICodeBase* codes,
	                        size_t n,
	                        const float zoom_y = 1.0f,
	                        bool reverse = false,
	                        std::function<void()> const& render_done = std::function<void()>()) = 0;
};

/**
 * Interface for synchronous backend
 *
 * TODO : Remove, it is not use anyway!!!
 */
class PVBCIDrawingBackendSync : public PVBCIDrawingBackend
{
  public:
	bool is_sync() const override { return true; }
};

/**
 * Interface for asynchronous Drawing Backend.
 */
class PVBCIDrawingBackendAsync : public PVBCIDrawingBackend
{
  public:
	bool is_sync() const override { return false; }

  public:
	// TODO : Remove this unused method.
	virtual void wait_all() const = 0;
};
}

#endif
