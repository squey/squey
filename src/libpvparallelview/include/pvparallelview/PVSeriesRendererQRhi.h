/* * MIT License
 *
 * © Squey, 2026
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

#ifndef _PVSERIESRENDERERQRHI_H_
#define _PVSERIESRENDERERQRHI_H_

#include <pvparallelview/PVSeriesAbstractRenderer.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace PVParallelView
{

class PVSeriesRhiContext;

/**
 * GPU renderer built on QRhi, Qt's own abstraction over Vulkan, Metal, Direct3D and
 * OpenGL. Going through QRhi rather than straight to Vulkan is what makes this one
 * backend instead of four, which is the whole reason the previous OpenGL renderer was
 * given up on.
 *
 * The picture is drawn into an off-screen texture and read back, rather than presented to
 * a window of its own. That costs a transfer per frame, and it is what buys the renderer
 * its plain place behind PVSeriesAbstractRenderer: no surface, no swapchain, no window
 * handle to keep in step with the widget, nothing to unwind when the view goes away.
 *
 * Only the sampled rows travel to the GPU; the geometry is derived from the vertex index
 * in the shader. Whether it beats PVSeriesRendererRaster depends entirely on the machine:
 * on a discrete card it wins by an order of magnitude, on a software rasteriser it loses,
 * which is why it is not the default and has to be asked for.
 */
class PVSeriesRendererQRhi : public PVSeriesAbstractRenderer
{
  public:
	explicit PVSeriesRendererQRhi(Squey::PVRangeSubSampler const& rss);
	~PVSeriesRendererQRhi() override;

	static bool capability();
	static PVSeriesView::DrawMode capability(PVSeriesView::DrawMode);

	void set_background_color(QColor const& bgcol) override;
	void set_draw_mode(PVSeriesView::DrawMode) override;

	QImage grab() override;

  private:
	PVSeriesView::DrawMode _draw_mode = PVSeriesView::DrawMode::Lines;
	QColor _background_color = Qt::black;

	std::unique_ptr<PVSeriesRhiContext> _context;
	std::vector<int16_t> _rows;
};

} // namespace PVParallelView

#endif // _PVSERIESRENDERERQRHI_H_
