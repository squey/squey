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

	enum class DrawMode { Lines, Points, LinesAlways, Default = Lines };

	enum class Backend { QPainter, OpenGL, OffscreenOpenGL, Default = OffscreenOpenGL };

	explicit PVSeriesView(Inendi::PVRangeSubSampler& rss,
	                      Backend backend = Backend::Default,
	                      QWidget* parent = 0);
	virtual ~PVSeriesView();

	void set_background_color(QColor const& bgcol);
	void show_series(std::vector<SerieDrawInfo> series_draw_order);
	void refresh();

	void set_draw_mode(DrawMode);
	static Backend capability(Backend);
	static DrawMode capability(Backend, DrawMode);

	template <class... Args>
	auto capability(Args&&... args)
	{
		return capability(_backend, std::forward<Args>(args)...);
	}

  protected:
	void paintEvent(QPaintEvent* event) override;
	void resizeEvent(QResizeEvent* event) override;

  private:
	Inendi::PVRangeSubSampler& _rss;
	std::unique_ptr<PVSeriesAbstractRenderer> _renderer;
	const Backend _backend;
	QPixmap _pixmap;
	bool _need_hard_redraw = false;

	auto make_renderer(Backend backend) -> Backend;
};

} // namespace PVParallelView

#endif // _PVSERIESVIEW_H_