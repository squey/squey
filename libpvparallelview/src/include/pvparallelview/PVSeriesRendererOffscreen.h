/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#ifndef _PVSERIESRENDEREROFFSCREEN_H_
#define _PVSERIESRENDEREROFFSCREEN_H_

#include <pvparallelview/PVSeriesRendererOpenGL.h>

#include <QOffscreenSurface>
#include <QDebug>

namespace PVParallelView
{

class PVSeriesRendererOffscreen : public PVSeriesAbstractRenderer, public QOffscreenSurface
{
  public:
	virtual ~PVSeriesRendererOffscreen() = default;

	void set_background_color(QColor const& bgcol) override
	{
		_gl_renderer.set_background_color(bgcol);
	}
	void resize(QSize const& size) override { _gl_renderer.resize(size); }
	QPixmap grab() override { return _gl_renderer.grab(); }
	void set_draw_mode(PVSeriesView::DrawMode mode) override { _gl_renderer.set_draw_mode(mode); }

	PVSeriesRendererOffscreen(Inendi::PVRangeSubSampler const& rss)
	    : PVSeriesAbstractRenderer(rss), QOffscreenSurface(), _gl_renderer(rss)
	{
		QSurfaceFormat format;
		format.setVersion(4, 3);
		format.setProfile(QSurfaceFormat::CoreProfile);
		setFormat(format);
		QOffscreenSurface::create();
		_gl_renderer.setFormat(QOffscreenSurface::format());
		qDebug() << "Could init QOffscreenSurface:" << isValid();
	}

	static bool capability()
	{
		static const bool s_offscreenopengl_capable = [] {
			QSurfaceFormat format;
			format.setVersion(4, 3);
			format.setProfile(QSurfaceFormat::CoreProfile);

			QOffscreenSurface offsc;
			offsc.setFormat(format);
			offsc.create();
			if (not offsc.isValid()) {
				qDebug() << "Imposible to create QOffscreenSurface";
				QOffscreenSurface offsc_crash;
				offsc_crash.create();
				if (not offsc_crash.isValid()) {
					qDebug() << "Absolutely impossible to create any QOffscreenSurface";
				}
				return false;
			}
			QOpenGLContext qogl;
			qogl.setFormat(offsc.format());
			return qogl.create() && qogl.format().version() >= qMakePair(4, 3);
		}();
		return s_offscreenopengl_capable;
	}

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