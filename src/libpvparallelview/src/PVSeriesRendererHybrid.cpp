//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvparallelview/PVSeriesRendererHybrid.h>

#include <squey/PVRangeSubSampler.h>

namespace PVParallelView
{

PVSeriesRendererHybrid::PVSeriesRendererHybrid(Squey::PVRangeSubSampler const& rss)
    : PVSeriesAbstractRenderer(rss), _raster(rss)
{
}

PVSeriesRendererHybrid::~PVSeriesRendererHybrid() = default;

void PVSeriesRendererHybrid::set_background_color(QColor const& bgcol)
{
	_background_color = bgcol;
	_raster.set_background_color(bgcol);
#ifdef SQUEY_SERIES_QRHI
	if (_qrhi) {
		_qrhi->set_background_color(bgcol);
	}
#endif
}

void PVSeriesRendererHybrid::set_draw_mode(PVSeriesView::DrawMode mode)
{
	_draw_mode = mode;
	_raster.set_draw_mode(mode);
#ifdef SQUEY_SERIES_QRHI
	if (_qrhi) {
		_qrhi->set_draw_mode(mode);
	}
#endif
}

void PVSeriesRendererHybrid::resize(QSize const& size)
{
	PVSeriesAbstractRenderer::resize(size);
	_raster.resize(size);
#ifdef SQUEY_SERIES_QRHI
	if (_qrhi) {
		_qrhi->resize(size);
	}
#endif
}

void PVSeriesRendererHybrid::on_show_series()
{
	// Both are kept in sync with every draw order, active or not: whichever becomes
	// active next must not render one frame behind on stale series.
	_raster.show_series(_series_draw_order);
#ifdef SQUEY_SERIES_QRHI
	if (_qrhi) {
		_qrhi->show_series(_series_draw_order);
	}
#endif
}

QImage PVSeriesRendererHybrid::grab()
{
#ifdef SQUEY_SERIES_QRHI
	const size_t total_work = size_t(_rss.samples_count()) * _series_draw_order.size();

	if (not _gpu_active and not _qrhi_unavailable and total_work >= high_threshold) {
		if (not _qrhi) {
			if (not PVSeriesRendererQRhi::capability()) {
				// No device answered : do not pay for a capability() re-probe on every
				// frame that crosses the threshold for the rest of the session.
				_qrhi_unavailable = true;
			} else {
				_qrhi = std::make_unique<PVSeriesRendererQRhi>(_rss);
				_qrhi->resize(_size);
				_qrhi->set_background_color(_background_color);
				_qrhi->set_draw_mode(_draw_mode);
				_qrhi->show_series(_series_draw_order);
			}
		}
		_gpu_active = bool(_qrhi);
	} else if (_gpu_active and total_work < low_threshold) {
		_gpu_active = false;
	}

	if (_gpu_active) {
		QImage image = _qrhi->grab();
		if (not image.isNull()) {
			return image;
		}
		// The device went away mid-session -- a driver reset, a laptop switching GPUs.
		// Fall back for good rather than retry every frame against a driver that just
		// failed.
		_gpu_active = false;
		_qrhi_unavailable = true;
	}
#endif
	return _raster.grab();
}

} // namespace PVParallelView
