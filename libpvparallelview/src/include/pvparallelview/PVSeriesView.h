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
#if QT_VERSION < QT_VERSION_CHECK(5, 4, 0)
#include <QGLWidget>
#include <QGLShaderProgram>
#include <QGLShader>
#include <QGLBuffer>
using PVOpenGLWidget = QGLWidget;
using QOpenGLShader = QGLShader;
using QOpenGLShaderProgram = QGLShaderProgram;
using QOpenGLBuffer = QGLBuffer;
#else
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLShader>
#include <QOpenGLBuffer>
using PVOpenGLWidget = QOpenGLWidget;
#endif

namespace PVParallelView
{

class PVSeriesView : public PVOpenGLWidget, protected QOpenGLFunctions
{
	Q_OBJECT

  public:
	explicit PVSeriesView(Inendi::PVRangeSubSampler& rss, QWidget* parent = 0);
	virtual ~PVSeriesView();

	void setBackgroundColor(QColor const& bgcol);

	void showAllSeries();
	void showSeries(std::vector<size_t> seriesDrawOrder);

	void onResampled();

  protected:
	void initializeGL() override;
	void cleanupGL();
	void resizeGL(int w, int h) override;
	void paintGL() override;

	void onAboutToCompose();
	void onFrameSwapped();

	void debugAvailableMemory();

	void compute_dbo_GL();

	void softPaintGL();

  private:
	struct Vertex {
		// GLfloat x;
		// GLfloat y;
		GLushort y;
	};

	Inendi::PVRangeSubSampler& m_rss;
	std::vector<size_t> m_seriesDrawOrder;

	QOpenGLVertexArrayObject m_vao;
	QOpenGLBuffer m_vbo;
	QOpenGLBuffer m_dbo;
	std::unique_ptr<QOpenGLShaderProgram> m_program;
	std::unique_ptr<QOpenGLFramebufferObject> m_fbo;
	std::unique_ptr<QOpenGLTexture> m_fbtexture;

	std::optional<QColor> m_backgroundColor;

	int m_verticesCount = 0;
	int m_linesPerVboCount = 0;
	size_t m_linesCount = 0;

	int m_batches = 1;

	int m_sizeLocation = 0;

	int m_w = 0, m_h = 0;

	bool m_needHardRedraw = true;

	bool m_wasCleanedUp = false;

	GLint m_GL_max_elements_vertices = 0;

	void (*glMultiDrawArraysIndirect)(GLenum mode,
	                                  const void* indirect,
	                                  GLsizei drawcount,
	                                  GLsizei stride) = nullptr;
	void* (*glMapBufferRange)(GLenum target,
	                          GLintptr offset,
	                          GLsizeiptr length,
	                          GLbitfield access) = nullptr;
	void (*glBlitFramebuffer)(GLint srcX0,
	                          GLint srcY0,
	                          GLint srcX1,
	                          GLint srcY1,
	                          GLint dstX0,
	                          GLint dstY0,
	                          GLint dstX1,
	                          GLint dstY1,
	                          GLbitfield mask,
	                          GLenum filter) = nullptr;

	std::chrono::time_point<std::chrono::high_resolution_clock> startCompositionTimer;
};

} // namespace PVParallelView

#endif // _PVSERIESVIEW_H_