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

#ifndef _PVSERIESRENDEREROFFSCREEN_H_
#define _PVSERIESRENDEREROFFSCREEN_H_

#include <pvparallelview/PVSeriesRendererOpenGL.h>

#include <QOffscreenSurface>

namespace PVParallelView
{

bool egl_support();
QString egl_vendor();
QString opengl_version();

class PVSeriesRendererOffscreen : public PVSeriesAbstractRenderer, public QOffscreenSurface
{
  public:
	constexpr static int OpenGLES_version_major = PVSeriesRendererOpenGL::OpenGLES_version_major,
	                     OpenGLES_version_minor = PVSeriesRendererOpenGL::OpenGLES_version_minor;

  public:
	PVSeriesRendererOffscreen(Inendi::PVRangeSubSampler const& rss);

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
