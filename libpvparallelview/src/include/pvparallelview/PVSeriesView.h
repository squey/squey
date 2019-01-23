/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2018
 */

#ifndef _PVSERIESVIEW_H_
#define _PVSERIESVIEW_H_

#include <inendi/PVRangeSubSampler.h>

#include <QOpenGLVertexArrayObject>
#include <QOpenGLFunctions>
#include <QtGlobal>
#include <QOpenGLTexture>
#include <QOpenGLFramebufferObject>
#include <QOpenGLWidget>
#include <QOpenGLExtraFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLShader>
#include <QOpenGLBuffer>

namespace PVParallelView
{

class PVSeriesViewCompat;

class PVSeriesView : public QOpenGLWidget, protected QOpenGLExtraFunctions
{
	Q_OBJECT

  public:
	struct SerieDrawInfo {
		size_t dataIndex;
		QColor color;
	};

  public:
	explicit PVSeriesView(Inendi::PVRangeSubSampler& rss, QWidget* parent = 0);
	virtual ~PVSeriesView();

	void setBackgroundColor(QColor const& bgcol);

	void showSeries(std::vector<SerieDrawInfo> seriesDrawOrder);

	void onResampled();

  protected:
	void initializeGL() override;
	void cleanupGL();
	void resizeGL(int w, int h) override;
	void paintGL() override;

	void paintEvent(QPaintEvent* event) override;
	void resizeEvent(QResizeEvent* event) override;

	void onAboutToCompose();
	void onFrameSwapped();

	void debugAvailableMemory();
	void debugErrors();

	void compute_dbo_GL();
	void fill_dbo_GL();
	void fill_vbo_GL(size_t const lineBegin, size_t const lineEnd);
	void fill_cbo_GL(size_t const lineBegin, size_t const lineEnd);
	void draw_GL(size_t const lineBegin, size_t const lineEnd);

	void setupShaders_GL();

	void softPaintGL();

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

	Inendi::PVRangeSubSampler& m_rss;
	std::vector<SerieDrawInfo> m_seriesDrawOrder;

	QOpenGLVertexArrayObject m_vao;
	QOpenGLBuffer m_vbo;
	QOpenGLBuffer m_cbo;
	QOpenGLBuffer m_dbo;
	std::unique_ptr<QOpenGLShaderProgram> m_program;
	std::unique_ptr<QOpenGLFramebufferObject> m_fbo;
	std::unique_ptr<QOpenGLTexture> m_fbtexture;

	QPixmap m_pixmap;
	std::unique_ptr<PVSeriesViewCompat> m_seriesViewCompat;

	std::optional<QColor> m_backgroundColor;

	int m_verticesCount = 0;
	int m_linesPerVboCount = 0;
	size_t m_linesCount = 0;

	int m_sizeLocation = 0;

	int m_w = 0, m_h = 0;

	bool m_needHardRedraw = true;

	bool m_wasCleanedUp = false;

	GLint m_GL_max_elements_vertices = 0;

	void (*glMultiDrawArraysIndirect)(GLenum mode,
	                                  const void* indirect,
	                                  GLsizei drawcount,
	                                  GLsizei stride) = nullptr;

	std::chrono::time_point<std::chrono::high_resolution_clock> startCompositionTimer;
};

} // namespace PVParallelView

#endif // _PVSERIESVIEW_H_