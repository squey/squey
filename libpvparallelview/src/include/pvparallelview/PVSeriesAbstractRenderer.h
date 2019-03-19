/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
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