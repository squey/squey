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

#ifndef _PVSERIESABSTRACTRENDERER_H_
#define _PVSERIESABSTRACTRENDERER_H_

#include <pvparallelview/PVSeriesView.h>

namespace PVParallelView
{

class PVSeriesAbstractRenderer
{
  public:
	virtual ~PVSeriesAbstractRenderer() = default;

	virtual void set_background_color(QColor const& bgcol) = 0;
	virtual void resize(QSize const& size) = 0;
	virtual QPixmap grab() = 0;

	virtual void set_draw_mode(PVSeriesView::DrawMode) = 0;

	void show_series(std::vector<PVSeriesView::SerieDrawInfo> seriesDrawOrder)
	{
		std::swap(_series_draw_order, seriesDrawOrder);
		on_show_series();
	}

  protected:
	PVSeriesAbstractRenderer(Inendi::PVRangeSubSampler const& rss) : _rss(rss) {}

	virtual void on_show_series() {}

	Inendi::PVRangeSubSampler const& _rss;
	std::vector<PVSeriesView::SerieDrawInfo> _series_draw_order;
};

} // namespace PVParallelView

#endif // _PVSERIESABSTRACTRENDERER_H_
