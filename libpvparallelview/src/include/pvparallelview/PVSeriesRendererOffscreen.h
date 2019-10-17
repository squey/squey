/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#ifndef _PVSERIESRENDEREROFFSCREEN_H_
#define _PVSERIESRENDEREROFFSCREEN_H_

#include <pvparallelview/PVSeriesRendererOpenGL.h>

#include <QOffscreenSurface>

namespace PVParallelView
{

class PVSeriesRendererOffscreen : public PVSeriesAbstractRenderer, public QOffscreenSurface
{
  public:
	constexpr static int OpenGLES_version_major = PVSeriesRendererOpenGL::OpenGLES_version_major,
	                     OpenGLES_version_minor = PVSeriesRendererOpenGL::OpenGLES_version_minor;

  public:
	PVSeriesRendererOffscreen(Inendi::PVRangeSubSampler const& rss);
	virtual ~PVSeriesRendererOffscreen();

	static bool capability();

	void set_background_color(QColor const& bgcol) override
	{
		_gl_renderer.set_background_color(bgcol);
	}
	void resize(QSize const& size) override { _gl_renderer.resize(size); }
	QPixmap grab() override { return _gl_renderer.grab(); }
	void set_draw_mode(PVSeriesView::DrawMode mode) override { _gl_renderer.set_draw_mode(mode); }

	template <class... Args>
	static auto capability(Args&&... args)
	{
		return PVSeriesRendererOpenGL::capability(std::forward<Args>(args)...);
	}

  protected:
	void on_show_series() override { _gl_renderer.show_series(std::move(_series_draw_order)); }

	PVSeriesRendererOpenGL _gl_renderer;
};

} // namespace PVParallelView

#endif // _PVSERIESRENDEREROFFSCREEN_H_
