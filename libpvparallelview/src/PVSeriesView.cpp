#include <pvparallelview/PVSeriesView.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <mutex>
#include <QCoreApplication>
#include <QPainter>
#include <QResizeEvent>
#include <QOffscreenSurface>

namespace PVParallelView
{

class PVSeriesAbstractRenderer
{
  public:
	virtual ~PVSeriesAbstractRenderer() = default;

	virtual void setBackgroundColor(QColor const& bgcol) = 0;
	virtual void resize(QSize const& size) = 0;
	virtual QPixmap grab() = 0;

	void showSeries(std::vector<PVSeriesView::SerieDrawInfo> seriesDrawOrder)
	{
		std::swap(m_seriesDrawOrder, seriesDrawOrder);
		onShowSeries();
	}

  protected:
	PVSeriesAbstractRenderer(Inendi::PVRangeSubSampler const& rss) : m_rss(rss) {}

	virtual void onShowSeries() {}

	Inendi::PVRangeSubSampler const& m_rss;
	std::vector<PVSeriesView::SerieDrawInfo> m_seriesDrawOrder;
};

class PVSeriesRendererQPainter : public PVSeriesAbstractRenderer, public QWidget
{

  public:
	PVSeriesRendererQPainter(Inendi::PVRangeSubSampler const& rss, QWidget* parent = nullptr)
	    : PVSeriesAbstractRenderer(rss), QWidget(parent)
	{
		setAutoFillBackground(true);
	}

	void setBackgroundColor(QColor const& bgcol) override { setPalette(QPalette(bgcol)); }

	void resize(QSize const& size) { return QWidget::resize(size); }
	QPixmap grab() { return QWidget::grab(); }

  protected:
	void paintEvent(QPaintEvent* event) override
	{
		QPainter painter(this);
		std::vector<QPoint> points;
		for (auto& serieDraw : m_seriesDrawOrder) {
			painter.setPen(serieDraw.color);
			auto& serieData = m_rss.averaged_timeserie(serieDraw.dataIndex);
			for (int j = 0; j < serieData.size();) {
				points.clear();
				while (j < serieData.size()) {
					int vertex = serieData[j];
					if (vertex & (1 << 15)) {     // if out of range
						if (vertex & (1 << 14)) { // if overflow
							vertex = (1 << 15);
						} else { // else underflow
							vertex = -(1 << 15);
						}
					} else if (vertex & (1 << 14)) { // else if no value
						while (serieData[++j] & (1 << 14))
							;
						break;
					}
					points.push_back(QPoint(j, height() - vertex * height() / (1 << 14)));
					++j;
				}
				if (j == serieData.size() && points.size() > 1) {
					qDebug() << points.back();
				}
				if (points.size() > 1) {
					painter.drawPolyline(points.data(), points.size());
				} else if (points.size() == 1) {
					painter.drawPoint(points.front());
				}
			}
		}
		painter.end();
	}
};

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
	virtual ~PVSeriesRendererOpenGL();

	static bool hasCapability();

	void setBackgroundColor(QColor const& bgcol) override;

	void resize(QSize const& size) override { QWidget::resize(size); }

	QPixmap grab() override { return QWidget::grab(); }

	void onShowSeries() override;

	void onResampled();

  protected:
	void initializeGL() override;
	void cleanupGL();
	void resizeGL(int w, int h) override;
	void paintGL() override;

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
	std::unique_ptr<QOpenGLShaderProgram> m_program;

	std::optional<QColor> m_backgroundColor;

	int m_linesPerVboCount = 0;

	int m_sizeLocation = 0;

	bool m_wasCleanedUp = false;

	GLint m_GL_max_elements_vertices = 0;

	void (*glMultiDrawArraysIndirect)(GLenum mode,
	                                  const void* indirect,
	                                  GLsizei drawcount,
	                                  GLsizei stride) = nullptr;

	std::chrono::time_point<std::chrono::high_resolution_clock> startCompositionTimer;
};

class PVSeriesRendererOffscreen : public PVSeriesAbstractRenderer,
                                  public QOffscreenSurface,
                                  public QWidget
{
  public:
	virtual ~PVSeriesRendererOffscreen() = default;

	virtual void setBackgroundColor(QColor const& bgcol) { m_glRenderer.setBackgroundColor(bgcol); }
	virtual void resize(QSize const& size) { m_glRenderer.resize(size); }
	virtual QPixmap grab() { return m_glRenderer.grab(); }

	PVSeriesRendererOffscreen(Inendi::PVRangeSubSampler const& rss)
	    : PVSeriesAbstractRenderer(rss)
	    , QOffscreenSurface()
	    , QWidget(nullptr)
	    , m_glRenderer(rss, this)
	{
		QSurfaceFormat format;
		format.setVersion(4, 3);
		format.setProfile(QSurfaceFormat::CoreProfile);
		setFormat(format);
		QOffscreenSurface::create();
		m_glRenderer.setFormat(QOffscreenSurface::format());
		qDebug() << "Could init QOffscreenSurface:" << isValid();
	}

	static bool hasCapability()
	{
		QSurfaceFormat format;
		format.setVersion(4, 3);
		format.setProfile(QSurfaceFormat::CoreProfile);

		QOffscreenSurface offsc;
		offsc.setFormat(format);
		offsc.create();
		if (not offsc.isValid()) {
			return false;
		}
		QOpenGLContext qogl;
		qogl.setFormat(offsc.format());
		return qogl.create() && qogl.format().version() >= qMakePair(4, 3);
	}

  protected:
	virtual void onShowSeries()
	{
		m_glRenderer.showSeries(std::move(m_seriesDrawOrder));
		m_glRenderer.onShowSeries();
	}

	PVSeriesRendererOpenGL m_glRenderer;
};

PVSeriesView::PVSeriesView(Inendi::PVRangeSubSampler& rss, QWidget* parent)
    : QWidget(parent), m_rss(rss)
{
	if (PVSeriesRendererOffscreen::hasCapability()) {
		qDebug() << "Choosing PVSeriesRendererOffscreen";
		m_renderer = std::make_unique<PVSeriesRendererOffscreen>(m_rss);
	} else if (PVSeriesRendererOpenGL::hasCapability()) {
		qDebug() << "Choosing PVSeriesRendererOpenGL";
		m_renderer = std::make_unique<PVSeriesRendererOpenGL>(m_rss);
	} else {
		qDebug() << "Choosing PVSeriesRendererQPainter";
		m_renderer = std::make_unique<PVSeriesRendererQPainter>(m_rss);
	}
}

PVSeriesView::~PVSeriesView() = default;

void PVSeriesView::setBackgroundColor(QColor const& bgcol)
{
	setPalette(QPalette(bgcol));
	m_renderer->setBackgroundColor(bgcol);
}

void PVSeriesView::showSeries(std::vector<SerieDrawInfo> seriesDrawOrder)
{
	m_renderer->showSeries(std::move(seriesDrawOrder));
	m_needHardRedraw = true;
	update();
}

void PVSeriesView::onResampled()
{
	m_needHardRedraw = true;
	update();
}

void PVSeriesView::paintEvent(QPaintEvent* event)
{
	if (m_needHardRedraw) {
		qDebug() << "hard paint";
		m_pixmap = m_renderer->grab();
		m_needHardRedraw = false;
	}
	qDebug() << "soft paint";
	QPainter painter(this);
	painter.drawPixmap(0, 0, m_pixmap);
}

void PVSeriesView::resizeEvent(QResizeEvent* event)
{
	qDebug() << "resize";
	m_rss.set_sampling_count(event->size().width());
	m_rss.resubsample();
	m_renderer->resize(event->size());
	m_needHardRedraw = true;
}

#define SHADER(x) #x
#define LOAD_GL_FUNC(x)                                                                            \
	x = reinterpret_cast<decltype(x)>(context()->getProcAddress(#x));                              \
	assert(x != nullptr);

PVSeriesRendererOpenGL::PVSeriesRendererOpenGL(Inendi::PVRangeSubSampler const& rss,
                                               QWidget* parent)
    : PVSeriesAbstractRenderer(rss)
    , QOpenGLWidget(parent)
    , m_dbo(static_cast<QOpenGLBuffer::Type>(GL_DRAW_INDIRECT_BUFFER))
{
	if (std::getenv("PVOPENGL45") != nullptr) {
		QSurfaceFormat format;
		format.setVersion(4, 5);
		format.setProfile(QSurfaceFormat::CoreProfile);
		setFormat(format);
	}

	// QSurfaceFormat format;
	// format.setVersion(3, 3);
	// format.setProfile(QSurfaceFormat::CoreProfile);
	// setFormat(format);

	// setUpdateBehavior(QOpenGLWidget::PartialUpdate);
}

PVSeriesRendererOpenGL::~PVSeriesRendererOpenGL()
{
	cleanupGL();
}

bool PVSeriesRendererOpenGL::hasCapability()
{
	QOpenGLContext qogl;
	QSurfaceFormat format;
	format.setVersion(4, 3);
	format.setProfile(QSurfaceFormat::CoreProfile);
	qogl.setFormat(format);
	return qogl.create() && qogl.format().version() >= qMakePair(4, 3);
}

void PVSeriesRendererOpenGL::debugAvailableMemory()
{
#define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049

	GLint total_mem_kb = 0;
	glGetIntegerv(GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX, &total_mem_kb);

	GLint cur_avail_mem_kb = 0;
	glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &cur_avail_mem_kb);

	qDebug() << cur_avail_mem_kb << " available on total " << total_mem_kb;
}

void PVSeriesRendererOpenGL::debugErrors()
{
#define PVCASE_GLERROR(val)                                                                        \
	case (val):                                                                                    \
		qDebug() << #val;                                                                          \
		break;
	GLenum err;
	while ((err = glGetError()) != GL_NO_ERROR) {
		qDebug() << "glGetError:" << err;
		switch (err) {
			PVCASE_GLERROR(GL_INVALID_ENUM)
			PVCASE_GLERROR(GL_INVALID_VALUE)
			PVCASE_GLERROR(GL_INVALID_OPERATION)
			PVCASE_GLERROR(GL_STACK_OVERFLOW)
			PVCASE_GLERROR(GL_STACK_UNDERFLOW)
			PVCASE_GLERROR(GL_OUT_OF_MEMORY)
		default:
			qDebug() << "Unknown_GL_error";
			break;
		}
	}
}

void PVSeriesRendererOpenGL::setBackgroundColor(QColor const& bgcol)
{
	m_backgroundColor = bgcol;
	setPalette(QPalette(bgcol));
}

void PVSeriesRendererOpenGL::onShowSeries()
{
	makeCurrent();
	compute_dbo_GL();
	doneCurrent();
}

void PVSeriesRendererOpenGL::initializeGL()
{
	connect(context(), &QOpenGLContext::aboutToBeDestroyed, this,
	        &PVSeriesRendererOpenGL::cleanupGL);
	// connect(this, &QOpenGLWidget::aboutToCompose, this,
	// &PVSeriesRendererOpenGL::onAboutToCompose);
	// connect(this, &QOpenGLWidget::frameSwapped, this, &PVSeriesRendererOpenGL::onFrameSwapped);

	initializeOpenGLFunctions();
	LOAD_GL_FUNC(glMultiDrawArraysIndirect);

	glGetIntegerv(GL_MAX_ELEMENTS_VERTICES, &m_GL_max_elements_vertices);
	qDebug() << "GL_MAX_ELEMENTS_VERTICES" << m_GL_max_elements_vertices;
	qDebug() << "Context:" << QString(reinterpret_cast<const char*>(glGetString(GL_VERSION)));
	qDebug() << "Screen:" << context()->screen();
	qDebug() << "Surface:" << context()->surface()->surfaceClass()
	         << context()->surface()->surfaceType()
	         << "(supports OpenGL:" << context()->surface()->supportsOpenGL() << ")"
	         << context()->surface()->size();

	auto start = std::chrono::system_clock::now();

	// glClearColor(1.0, 0.2, 0.8, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);

	setupShaders_GL();

	m_vao.create();
	m_vbo.create();
	m_cbo.create();
	m_dbo.create();

	m_vao.bind();
	m_program->bind();

	m_sizeLocation = m_program->uniformLocation("size");

	{
		m_vbo.bind();
		m_vbo.setUsagePattern(QOpenGLBuffer::StreamDraw);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 1, GL_UNSIGNED_SHORT, GL_FALSE, 0, 0x0);

		m_vbo.release();
	}

	{
		m_cbo.bind();
		m_cbo.setUsagePattern(QOpenGLBuffer::StreamDraw);

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0x0);
		glVertexAttribDivisor(1, 1);

		m_cbo.release();
	}

	m_vao.release();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	qDebug() << "initialiseGL:" << diff.count();

	debugErrors();

	if (m_wasCleanedUp) {
		m_wasCleanedUp = false;
		resizeGL(width(), height());
		update();
	}
}

void PVSeriesRendererOpenGL::cleanupGL()
{
	if (m_wasCleanedUp) {
		return;
	}
	qDebug() << "cleanupGL";
	makeCurrent();
	m_program.reset();
	m_dbo.destroy();
	m_cbo.destroy();
	m_vbo.destroy();
	m_vao.destroy();
	doneCurrent();
	m_wasCleanedUp = true;
}

void PVSeriesRendererOpenGL::resizeGL(int w, int h)
{
	qDebug() << "resizeGL(" << w << ", " << h << ")";
	m_vao.bind();

	glViewport(0, 0, w, h);

	compute_dbo_GL();

	m_program->setUniformValue(m_sizeLocation, QVector4D(0, 0, 0, w));

	m_vao.release();

	// debugAvailableMemory();
	debugErrors();
}

void PVSeriesRendererOpenGL::paintGL()
{
	glViewport(0, 0, width(), height());
	auto start = std::chrono::system_clock::now();

	m_vao.bind();

	if (m_backgroundColor) {
		glClearColor(m_backgroundColor->redF(), m_backgroundColor->greenF(),
		             m_backgroundColor->blueF(), m_backgroundColor->alphaF());
		m_backgroundColor = std::nullopt;
	}

	glClear(GL_COLOR_BUFFER_BIT);

	// glEnable(GL_MULTISAMPLE);
	// glEnable(GL_LINE_SMOOTH);
	// glLineWidth(3.f);

	for (size_t line = 0; line < m_seriesDrawOrder.size();) {
		auto lineEnd = std::min(line + m_linesPerVboCount, m_seriesDrawOrder.size());
		fill_vbo_GL(line, lineEnd);
		fill_cbo_GL(line, lineEnd);
		draw_GL(line, lineEnd);
		line = lineEnd;
	}

	m_vao.release();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	qDebug() << "paintGL:" << diff.count();

	debugErrors();
}

void PVSeriesRendererOpenGL::compute_dbo_GL()
{
	auto lines_count = m_seriesDrawOrder.size();

	if (lines_count <= 0 or width() <= 0) {
		return;
	}

	m_linesPerVboCount =
	    std::min(static_cast<size_t>(m_GL_max_elements_vertices / width()), lines_count);

	fill_dbo_GL();

	m_vbo.bind();
	m_vbo.allocate(m_linesPerVboCount * width() * sizeof(Vertex));
	assert(m_vbo.size() > 0);
	m_vbo.release();

	m_cbo.bind();
	m_cbo.allocate(m_linesPerVboCount * sizeof(CboBlock));
	assert(m_cbo.size() > 0);
	m_cbo.release();
}

void PVSeriesRendererOpenGL::fill_dbo_GL()
{
	m_dbo.bind();

	m_dbo.allocate(m_linesPerVboCount * sizeof(DrawArraysIndirectCommand));
	assert(m_dbo.size() > 0);
	DrawArraysIndirectCommand* dbo_bytes =
	    static_cast<DrawArraysIndirectCommand*>(m_dbo.map(QOpenGLBuffer::WriteOnly));
	assert(dbo_bytes);
	std::generate(dbo_bytes, dbo_bytes + m_linesPerVboCount, [ line = 0, this ]() mutable {
		size_t drawIndex = line++;
		return DrawArraysIndirectCommand{GLuint(width()), 1, GLuint(drawIndex * width()),
		                                 drawIndex};
	});

	m_dbo.unmap();
	m_dbo.release();
}

void PVSeriesRendererOpenGL::fill_vbo_GL(size_t const lineBegin, size_t const lineEnd)
{
	size_t const line_byte_size = width() * sizeof(Vertex);

	m_vbo.bind();
	void* vbo_bytes = glMapBufferRange(GL_ARRAY_BUFFER, 0, m_vbo.size(),
	                                   GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT |
	                                       GL_MAP_UNSYNCHRONIZED_BIT);
	assert(vbo_bytes);
	for (size_t line = lineBegin; line < lineEnd; ++line) {
		std::memcpy(reinterpret_cast<uint8_t*>(vbo_bytes) + (line - lineBegin) * line_byte_size,
		            reinterpret_cast<uint8_t const*>(
		                m_rss.averaged_timeserie(m_seriesDrawOrder[line].dataIndex).data()),
		            line_byte_size);
	}
	m_vbo.unmap();
	m_vbo.release();
}

void PVSeriesRendererOpenGL::fill_cbo_GL(size_t const lineBegin, size_t const lineEnd)
{
	m_cbo.bind();
	CboBlock* cbo_bytes = static_cast<CboBlock*>(glMapBufferRange(
	    GL_ARRAY_BUFFER, 0, m_cbo.size(),
	    GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT));
	assert(cbo_bytes);
	for (size_t line = lineBegin; line < lineEnd; ++line) {
		cbo_bytes[line - lineBegin].colorR = m_seriesDrawOrder[line].color.redF();
		cbo_bytes[line - lineBegin].colorG = m_seriesDrawOrder[line].color.greenF();
		cbo_bytes[line - lineBegin].colorB = m_seriesDrawOrder[line].color.blueF();
	}
	m_cbo.unmap();
	m_cbo.release();
}

void PVSeriesRendererOpenGL::draw_GL(size_t const lineBegin, size_t const lineEnd)
{
	m_dbo.bind();
	glMultiDrawArraysIndirect(GL_LINE_STRIP, 0, lineEnd - lineBegin,
	                          sizeof(DrawArraysIndirectCommand));
	m_dbo.release();

	// for (size_t line = 0; line < lineEnd - lineBegin; ++line) {
	// 	m_program->setUniformValue(m_sizeLocation, QVector4D(m_seriesDrawOrder[line].color.redF(),
	// 	                                                     m_seriesDrawOrder[line].color.greenF(),
	// 	                                                     m_seriesDrawOrder[line].color.blueF(),
	// 	                                                     width()));
	// 	glDrawArrays(GL_LINE_STRIP, line * width(), width());
	// }
}

// clang-format off
void PVSeriesRendererOpenGL::setupShaders_GL()
{
	m_program = std::make_unique<QOpenGLShaderProgram>(context());

	std::string vertexShader =
"#version 430\n" SHADER(
in vec4 vertex;
in vec3 color;
out vec4 lineColor;
uniform vec4 size;
void main(void) {
    //float line = floor(gl_VertexID / size.w);
    //float line = gl_InstanceID;
    //lineColor = vec4(fract((line + 1) * 0.7), fract(2 * (line + 1) * 0.7), fract(3 * (line + 1) * 0.7), 1);
    lineColor = vec4(color, 1);
    //lineColor = vec4(size.rgb, 1);
    vec4 wvertex = vertex;
    wvertex.y = vertex.x / 16383;// ((1 << 14) - 1);
    wvertex.x = mod(gl_VertexID, size.w) / (size.w - 1);
    int vx = int(vertex.x);
    if(bool(vx & (1 << 15))) { //if out of range
    	if(bool(vx & (1 << 14))) { //if overflow
    		wvertex.y = 1.5;
    	} else { //else underflow
    		wvertex.y = -0.5;
    	}
    } else if(bool(vx & (1 << 14))) { //else if no value
    	lineColor = vec4(0.,0.,0.,0.);
    	wvertex.y = 0.5;
    }
    gl_Position.xy = vec2(fma(wvertex.x, 2, -1), fma(wvertex.y, 2, -1));
});

	std::string fragmentShader =
"#version 430\n" SHADER(
in vec4 lineColor;
out vec4 FragColor;
void main(void) {
	FragColor = lineColor*floor(lineColor.a);
	// FragColor = vec4(1,1,1,1);
});

	m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShader.c_str());
	m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShader.c_str());

	m_program->link();
	qDebug() << m_program->log();
}
// clang-format on

void PVSeriesRendererOpenGL::onAboutToCompose()
{
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end - startCompositionTimer;
	qDebug() << "compositionGL interval:" << diff.count();
	startCompositionTimer = std::chrono::high_resolution_clock::now();
}

void PVSeriesRendererOpenGL::onFrameSwapped()
{
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end - startCompositionTimer;
	qDebug() << "compositionGL:" << diff.count();
}

} // namespace PVParallelView
