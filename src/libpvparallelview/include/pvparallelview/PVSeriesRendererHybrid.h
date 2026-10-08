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

#ifndef _PVSERIESRENDERERHYBRID_H_
#define _PVSERIESRENDERERHYBRID_H_

#include <pvparallelview/PVSeriesAbstractRenderer.h>
#include <pvparallelview/PVSeriesRendererRaster.h>

#ifdef SQUEY_SERIES_QRHI
#include <pvparallelview/PVSeriesRendererQRhi.h>
#endif

#include <memory>

namespace PVParallelView
{

/**
 * Picks the CPU rasteriser or the GPU renderer, per frame, from how much work the frame
 * actually is: samples shown times series drawn. Neither backend wins outright -- the
 * GPU's fixed per-frame round trip (upload, submit, wait, read back) only pays for itself
 * past a few hundred series -- and grouping or splitting a column is exactly what can
 * push a view across that line in the middle of a session.
 *
 * The GPU renderer, once it has proven it can run at all, is kept alive for the rest of
 * the session rather than rebuilt on every crossing: standing up a device and its
 * pipeline dwarfs the cost of a frame, reusing an idle one does not. Two thresholds
 * instead of one keep a workload sitting near the line from flipping the picture every
 * other frame.
 */
class PVSeriesRendererHybrid : public PVSeriesAbstractRenderer
{
  public:
	explicit PVSeriesRendererHybrid(Squey::PVRangeSubSampler const& rss);
	~PVSeriesRendererHybrid() override;

	static constexpr bool capability() { return true; }
	static PVSeriesView::DrawMode capability(PVSeriesView::DrawMode mode)
	{
		return PVSeriesRendererRaster::capability(mode);
	}

	void set_background_color(QColor const& bgcol) override;
	void set_draw_mode(PVSeriesView::DrawMode) override;
	void resize(QSize const& size) override;

	QImage grab() override;

	// Which of the two backends the last grab() actually used : for a status line, a log,
	// or a test confirming the switch fires rather than only that the picture is right.
	bool used_gpu_last_frame() const { return _gpu_active; }

	// samples_count() x series drawn, above which the GPU has beaten the CPU rasteriser
	// on every card measured so far -- an Intel iGPU (7.5ms vs 8.7ms) and a discrete RTX
	// 3060 (4.2ms vs 7.7ms), both right at this point. Below low_threshold it lost.
	static constexpr size_t high_threshold = 256 * 1024;
	static constexpr size_t low_threshold = high_threshold * 65 / 100;

  protected:
	void on_show_series() override;

  private:
	QColor _background_color = Qt::black;
	PVSeriesView::DrawMode _draw_mode = PVSeriesView::DrawMode::Lines;

	PVSeriesRendererRaster _raster;
#ifdef SQUEY_SERIES_QRHI
	std::unique_ptr<PVSeriesRendererQRhi> _qrhi;
	bool _qrhi_unavailable = false;
#endif
	bool _gpu_active = false;
};

} // namespace PVParallelView

#endif // _PVSERIESRENDERERHYBRID_H_
