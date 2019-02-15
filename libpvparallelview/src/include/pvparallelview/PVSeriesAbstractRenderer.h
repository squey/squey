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

	virtual void setBackgroundColor(QColor const& bgcol) = 0;
	virtual void resize(QSize const& size) = 0;
	virtual QPixmap grab() = 0;

	virtual void setDrawMode(PVSeriesView::DrawMode) = 0;

	void showSeries(std::vector<PVSeriesView::SerieDrawInfo> seriesDrawOrder)
	{
		std::swap(m_seriesDrawOrder, seriesDrawOrder);
		onShowSeries();
	}

  protected:
	PVSeriesAbstractRenderer(Inendi::PVRangeSubSampler const& rss) : m_rss(rss) {}

	virtual void onShowSeries() {}

	Inendi::PVRangeSubSampler const& m_rss;
	std::vector<PVSeriesView::SerieDrawInfo> m_seriesDrawOrder;
};

} // namespace PVParallelView

#endif // _PVSERIESABSTRACTRENDERER_H_