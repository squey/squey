/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#ifndef PVPARALLELVIEW_PVBCIDRAWINGBACKENDQPAINTER_H
#define PVPARALLELVIEW_PVBCIDRAWINGBACKENDQPAINTER_H

#include <pvkernel/opencl/common.h>

#include <pvparallelview/PVBCIDrawingBackend.h>

#include <map>
#include <vector>

namespace PVParallelView
{

class PVBCIBackendImageQPainter;

class PVBCIDrawingBackendQPainter : public PVBCIDrawingBackendAsync
{
	using backend_image_t = PVBCIBackendImageQPainter;

  public:
	~PVBCIDrawingBackendQPainter() override = default;

  public:
	bool is_gpu_accelerated() const override { return false; }

  public:
	static PVBCIDrawingBackendQPainter& get();

  public:
	backend_image_p_t create_image(size_t img_width, uint8_t height_bits) override;
	Flags flags() const override { return Serial; }

  public:
	PVBCICodeBase* allocate_bci(size_t n) override
	{
		return (PVBCICodeBase*)PVBCICode<>::allocate_codes(n);
	}
	void free_bci(PVBCICodeBase* buf) override
	{
		return PVBCICode<>::free_codes((PVBCICode<>*)buf);
	}

  public:
	// If this backend is synchronous, render_done must be ignored.
	void render(PVBCIBackendImage_p& dst_img,
	            size_t x_start,
	            size_t width,
	            PVBCICodeBase* codes,
	            size_t n,
	            const float zoom_y = 1.0f,
	            bool reverse = false,
	            std::function<void()> const& render_done = std::function<void()>()) override;

	void wait_all() const override {}

	std::shared_ptr<backend_image_t> _backend_image;
};

} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVBCIDRAWINGBACKENDQPAINTER_H
