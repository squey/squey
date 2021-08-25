/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
