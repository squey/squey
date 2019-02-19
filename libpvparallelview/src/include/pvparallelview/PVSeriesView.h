/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2018
 */

#ifndef _PVSERIESVIEW_H_
#define _PVSERIESVIEW_H_

#include <inendi/PVRangeSubSampler.h>

#include <QWidget>

namespace PVParallelView
{

class PVSeriesAbstractRenderer;

class PVSeriesView : public QWidget
{
	Q_OBJECT

  public:
	struct SerieDrawInfo {
		size_t dataIndex;
		QColor color;
	};

	enum class DrawMode { Lines, Points, Default = Lines };

	enum class Backend { QPainter, OpenGL, OffscreenOpenGL, Default = OffscreenOpenGL };

	explicit PVSeriesView(Inendi::PVRangeSubSampler& rss,
	                      Backend backend = Backend::Default,
	                      QWidget* parent = 0);
	virtual ~PVSeriesView();

	void setBackgroundColor(QColor const& bgcol);
	void showSeries(std::vector<SerieDrawInfo> seriesDrawOrder);
	void refresh();

	void setDrawMode(DrawMode);
	static Backend capability(Backend);
	static DrawMode capability(Backend, DrawMode);

	template <class... Args>
	auto capability(Args&&... args)
	{
		return capability(m_backend, std::forward<Args>(args)...);
	}

  protected:
	void paintEvent(QPaintEvent* event) override;
	void resizeEvent(QResizeEvent* event) override;

  private:
	Inendi::PVRangeSubSampler& m_rss;
	std::unique_ptr<PVSeriesAbstractRenderer> m_renderer;
	const Backend m_backend;
	QPixmap m_pixmap;
	bool m_needHardRedraw = false;

	auto make_renderer(Backend backend) -> Backend;
};

} // namespace PVParallelView

#endif // _PVSERIESVIEW_H_