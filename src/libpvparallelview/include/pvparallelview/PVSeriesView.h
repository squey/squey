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

#ifndef _PVSERIESVIEW_H_
#define _PVSERIESVIEW_H_

#include <squey/PVRangeSubSampler.h>

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

	enum class Backend { QPainter, OpenGL, OffscreenOpenGL, Default = QPainter };

	explicit PVSeriesView(Squey::PVRangeSubSampler& rss,
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
	Squey::PVRangeSubSampler& _rss;
	std::unique_ptr<PVSeriesAbstractRenderer> _renderer;
	const Backend _backend;
	QPixmap _pixmap;
	bool _need_hard_redraw = false;

	auto make_renderer(Backend backend) -> Backend;
};

} // namespace PVParallelView

#endif // _PVSERIESVIEW_H_
