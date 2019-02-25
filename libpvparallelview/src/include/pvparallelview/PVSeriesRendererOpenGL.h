/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
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

  public:
	struct SerieDrawInfo {
		size_t dataIndex;
		QColor color;
	};

  public:
	explicit PVSeriesRendererOpenGL(Inendi::PVRangeSubSampler const& rss, QWidget* parent = 0);
	virtual ~PVSeriesRendererOpenGL() noexcept;

	static bool capability();
	static PVSeriesView::DrawMode capability(PVSeriesView::DrawMode);

	void setBackgroundColor(QColor const& bgcol) override;
	void setDrawMode(PVSeriesView::DrawMode) override;

	void resize(QSize const& size) override { QWidget::resize(size); }

	QPixmap grab() override { return QWidget::grab(QRect(QPoint(0, 0), size())); }

	void onResampled();

  protected:
	void initializeGL() override;
	void cleanupGL();
	void resizeEvent(QResizeEvent* event) override;
	void resizeGL(int w, int h) override;
	void paintGL() override;

	void onAboutToCompose();
	void onFrameSwapped();

	void debugAvailableMemory();
	void debugErrors();

	void setDrawMode_GL();

	int lines_per_vbo() const;
	void compute_dbo_GL();
	void allocate_buffer_GL(QOpenGLBuffer& buffer, int expected_size);
	void fill_dbo_GL();
	void fill_vbo_GL(size_t const lineBegin, size_t const lineEnd);
	void fill_cbo_GL(size_t const lineBegin, size_t const lineEnd);
	void draw_GL(size_t const lineBegin, size_t const lineEnd);

	void setupShaders_GL();

  private:
	struct Vertex {
		// GLfloat x;
		// GLfloat y;
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

	QOpenGLVertexArrayObject m_vao;
	QOpenGLBuffer m_vbo;
	QOpenGLBuffer m_cbo;
	QOpenGLBuffer m_dbo;
	QOpenGLShaderProgram* m_program = nullptr;
	std::unique_ptr<QOpenGLShaderProgram> m_programLines;
	std::unique_ptr<QOpenGLShaderProgram> m_programPoints;
	std::unique_ptr<QOpenGLShaderProgram> m_programLinesAlways;

	std::optional<QColor> m_backgroundColor;
	bool m_needToResetDrawMode = false;
	PVSeriesView::DrawMode m_drawMode;
	GLenum m_glDrawMode = GL_LINE_STRIP;

	int m_linesPerVboCount = 0;

	int m_sizeLocation = 0;

	bool m_wasCleanedUp = false;
	bool m_blockPaint = false;
	QSize m_oldSize;

	GLint m_GL_max_elements_vertices = 0;

	void (*glMultiDrawArraysIndirect)(GLenum mode,
	                                  const void* indirect,
	                                  GLsizei drawcount,
	                                  GLsizei stride) = nullptr;

	std::chrono::time_point<std::chrono::high_resolution_clock> startCompositionTimer;
};

} // namespace PVParallelView

#endif // _PVSERIESRENDEREROPENGL_H_