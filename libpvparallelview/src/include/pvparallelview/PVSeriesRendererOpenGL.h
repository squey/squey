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

#ifndef _PVSERIESRENDEREROPENGL_H_
#define _PVSERIESRENDEREROPENGL_H_

#include <pvparallelview/PVSeriesAbstractRenderer.h>

#include <QOpenGLWidget>
#include <QOpenGLExtraFunctions>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QOpenGLShaderProgram>

namespace PVParallelView
{

class PVSeriesRendererOpenGL : public PVSeriesAbstractRenderer,
                               public QOpenGLWidget,
                               protected QOpenGLExtraFunctions
{
	using PVRSS = Inendi::PVRangeSubSampler;

  public:
	constexpr static int OpenGLES_version_major = 3, OpenGLES_version_minor = 2;

	struct SerieDrawInfo {
		size_t dataIndex;
		QColor color;
	};

  public:
	explicit PVSeriesRendererOpenGL(Inendi::PVRangeSubSampler const& rss, QWidget* parent = 0);
	virtual ~PVSeriesRendererOpenGL() noexcept;

	static bool capability();
	static PVSeriesView::DrawMode capability(PVSeriesView::DrawMode);

	void set_background_color(QColor const& bgcol) override;
	void set_draw_mode(PVSeriesView::DrawMode) override;

	void resize(QSize const& size) override { QWidget::resize(size); }

	QPixmap grab() override { return QWidget::grab(QRect(QPoint(0, 0), size())); }

	void on_resampled();

  protected:
	void initializeGL() override;
	void cleanupGL();
	void resizeEvent(QResizeEvent* event) override;
	void resizeGL(int w, int h) override;
	void paintGL() override;

	void debug_available_memory();
	void debug_errors();

	void set_draw_mode_GL();

	int lines_per_vbo() const;
	void compute_dbo_GL();
	void allocate_buffer_GL(QOpenGLBuffer& buffer, int expected_size);
	void fill_dbo_GL();
	void fill_vbo_GL(size_t const line_begin, size_t const line_end);
	void fill_cbo_GL(size_t const line_begin, size_t const line_end);
	void draw_GL(size_t const line_begin, size_t const line_end);

	void setup_shaders_GL();

  private:
	struct Vertex {
		GLushort y;
	};

	struct CboBlock {
		GLfloat colorR;
		GLfloat colorG;
		GLfloat colorB;
	};

	struct DrawArraysIndirectCommand {
		GLuint count;
		GLuint instanceCount;
		GLuint first;
		GLuint baseInstance;
	};

	QOpenGLVertexArrayObject _vao;
	QOpenGLBuffer _vbo;
	QOpenGLBuffer _cbo;
	QOpenGLBuffer _dbo;
	QOpenGLShaderProgram* _program = nullptr;
	std::unique_ptr<QOpenGLShaderProgram> _program_Lines;
	std::unique_ptr<QOpenGLShaderProgram> _program_Points;
	std::unique_ptr<QOpenGLShaderProgram> _program_LinesAlways;

	std::optional<QColor> _background_color;
	bool _need_to_reset_draw_mode = false;
	PVSeriesView::DrawMode _draw_mode;
	GLenum _gl_draw_mode = GL_LINE_STRIP;

	int _lines_per_vbo_count = 0;

	int _size_location = 0;

	bool _was_cleaned_up = false;
	bool _block_paint = false;
	QSize _old_size;

	GLint _GL_max_elements_vertices = 0;

	void (*glMultiDrawArraysIndirect)(GLenum mode,
	                                  const void* indirect,
	                                  GLsizei drawcount,
	                                  GLsizei stride) = nullptr;
};

} // namespace PVParallelView

#endif // _PVSERIESRENDEREROPENGL_H_
