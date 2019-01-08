#include <pvparallelview/PVSeriesView.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <mutex>
#include <QCoreApplication>
#include <QPainter>

namespace PVParallelView
{

#define SHADER(x) #x
#define LOAD_GL_FUNC(x)                                                                            \
	x = reinterpret_cast<decltype(x)>(context()->getProcAddress(#x));                              \
	assert(x != nullptr);

PVSeriesView::PVSeriesView(Inendi::PVRangeSubSampler& rss, QWidget* parent)
    : PVOpenGLWidget(parent)
    , m_rss(rss)
    , m_dbo(static_cast<QOpenGLBuffer::Type>(GL_DRAW_INDIRECT_BUFFER))
{
	QCoreApplication::setAttribute(Qt::AA_DontCreateNativeWidgetSiblings);
	qDebug() << "Qt::AA_DontCreateNativeWidgetSiblings:"
	         << QCoreApplication::testAttribute(Qt::AA_DontCreateNativeWidgetSiblings);

	if (std::getenv("PVOPENGL45") != nullptr) {
		QSurfaceFormat format;
		format.setVersion(4, 5);
		format.setProfile(QSurfaceFormat::CoreProfile);
		setFormat(format);
	}

	// setUpdateBehavior(QOpenGLWidget::PartialUpdate);
}

PVSeriesView::~PVSeriesView()
{
	cleanupGL();
}

void PVSeriesView::debugAvailableMemory()
{
#define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049

	GLint total_mem_kb = 0;
	glGetIntegerv(GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX, &total_mem_kb);

	GLint cur_avail_mem_kb = 0;
	glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &cur_avail_mem_kb);

	qDebug() << cur_avail_mem_kb << " available on total " << total_mem_kb;
}

void PVSeriesView::setBackgroundColor(QColor const& bgcol)
{
	m_backgroundColor = bgcol;
}

void PVSeriesView::showSeries(std::vector<PVSeriesView::SerieDrawInfo> seriesDrawOrder)
{
	std::swap(m_seriesDrawOrder, seriesDrawOrder);

	makeCurrent();

	compute_dbo_GL();

	doneCurrent();

	m_needHardRedraw = true;
}

void PVSeriesView::onResampled()
{
	m_needHardRedraw = true;
	update();
}

void PVSeriesView::initializeGL()
{
	connect(context(), &QOpenGLContext::aboutToBeDestroyed, this, &PVSeriesView::cleanupGL);
	// connect(this, &QOpenGLWidget::aboutToCompose, this, &PVSeriesView::onAboutToCompose);
	// connect(this, &QOpenGLWidget::frameSwapped, this, &PVSeriesView::onFrameSwapped);

	initializeOpenGLFunctions();
	LOAD_GL_FUNC(glMultiDrawArraysIndirect);
	LOAD_GL_FUNC(glMapBufferRange);
	LOAD_GL_FUNC(glBlitFramebuffer);
	LOAD_GL_FUNC(glVertexAttribDivisor);

	glGetIntegerv(GL_MAX_ELEMENTS_VERTICES, &m_GL_max_elements_vertices);

	qDebug() << "Context:" << QString(reinterpret_cast<const char*>(glGetString(GL_VERSION)));

	auto start = std::chrono::system_clock::now();

	// glClearColor(1.0, 0.2, 0.8, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);

	m_program = std::make_unique<QOpenGLShaderProgram>(context());

	// clang-format off
	m_program->addShaderFromSourceCode(QOpenGLShader::Vertex,
"#version 450\n" SHADER(
in vec4 vertex;
in vec3 color;
out vec4 lineColor;
uniform vec4 size;
void main(void) {
    //float line = floor(gl_VertexID / size.w);
    //float line = gl_InstanceID;
    //lineColor = vec4(fract((line + 1) * 0.7), fract(2 * (line + 1) * 0.7), fract(3 * (line + 1) * 0.7), 1);
    lineColor = vec4(color, 1);
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
}));
	m_program->addShaderFromSourceCode(QOpenGLShader::Fragment,
"#version 450\n" SHADER(
in vec4 lineColor;
out vec4 FragColor;
void main(void) {
	FragColor = lineColor*floor(lineColor.a);
	// FragColor = vec4(1,1,1,1);
}));
	// clang-format on

	m_program->link();
	qDebug() << m_program->log();

	m_vao.create();
	m_vbo.create();
	m_cbo.create();
	m_dbo.create();

	m_vao.bind();
	m_program->bind();

	m_sizeLocation = m_program->uniformLocation("size");

	m_vbo.bind();
	m_vbo.setUsagePattern(QOpenGLBuffer::StreamDraw);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 1, GL_UNSIGNED_SHORT, GL_FALSE, 0, 0x0);

	m_vbo.release();

	m_cbo.bind();
	m_vbo.setUsagePattern(QOpenGLBuffer::StreamDraw);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0x0);
	glVertexAttribDivisor(1, 1);

	m_cbo.release();

	m_vao.release();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	qDebug() << "initialiseGL:" << diff.count();

	resizeGL(m_w, m_h);

	if (m_wasCleanedUp) {
		m_wasCleanedUp = false;
		update();
	}
}

void PVSeriesView::cleanupGL()
{
	if (m_wasCleanedUp) {
		return;
	}
	qDebug() << "cleanupGL";
	makeCurrent();
	m_fbtexture->destroy();
	m_fbtexture.reset();
	m_fbo.reset();
	m_program.reset();
	m_dbo.destroy();
	m_cbo.destroy();
	m_vbo.destroy();
	m_vao.destroy();
	doneCurrent();
	m_wasCleanedUp = true;
}

void PVSeriesView::resizeGL(int w, int h)
{
	qDebug() << "resizeGL(" << w << ", " << h << ")";
	m_vao.bind();

	if (w != m_w or h != m_h) {
		m_w = w;
		m_h = h;
		glViewport(0, 0, w, h);
		// glFrustum(0,0,0,0,0.5,1);

		m_rss.set_sampling_count(w);
		m_rss.resubsample();
		m_verticesCount = m_rss.samples_count();

		m_fbo = std::make_unique<QOpenGLFramebufferObject>(m_w, m_h);
	}

	compute_dbo_GL();

	m_program->setUniformValue(m_sizeLocation, QVector4D(0, 0, 0, m_verticesCount));

	m_vao.release();

	m_needHardRedraw = true;
}

void PVSeriesView::paintGL()
{
	if (not m_needHardRedraw) {
		softPaintGL();
		return;
	}

	glViewport(0, 0, m_w, m_h);
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

	m_dbo.bind();

	for (size_t line = 0; line < m_seriesDrawOrder.size();) {
		auto lineEnd = std::min(line + m_linesPerVboCount, m_seriesDrawOrder.size());
		fill_vbo_GL(line, lineEnd);
		fill_cbo_GL(line, lineEnd);
		line = lineEnd;

		for (int b = 0; b < m_batches; ++b) {
			glMultiDrawArraysIndirect(
			    GL_LINE_STRIP, reinterpret_cast<void*>(b * (m_linesPerVboCount / m_batches) *
			                                           sizeof(DrawArraysIndirectCommand)),
			    m_linesPerVboCount / m_batches, sizeof(DrawArraysIndirectCommand));
			// qDebug() << glGetError() << m_linesPerVboCount / m_batches;
		}
		if (m_linesPerVboCount % m_batches) {
			glMultiDrawArraysIndirect(
			    GL_LINE_STRIP,
			    reinterpret_cast<void*>(m_batches * (m_linesPerVboCount / m_batches) *
			                            sizeof(DrawArraysIndirectCommand)),
			    m_linesPerVboCount % m_batches, sizeof(DrawArraysIndirectCommand));
			// qDebug() << glGetError();
		}
	}

	m_dbo.release();
	m_vao.release();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	qDebug() << "paintGL:" << diff.count();

	m_fbtexture = std::make_unique<QOpenGLTexture>(grabFramebuffer().mirrored(),
	                                               QOpenGLTexture::DontGenerateMipMaps);

	m_needHardRedraw = false;

	// debugAvailableMemory();
}

void PVSeriesView::compute_dbo_GL()
{
	auto lines_count = m_seriesDrawOrder.size();

	if (lines_count <= 0 or m_verticesCount <= 0) {
		return;
	}

	m_linesPerVboCount =
	    std::min(static_cast<size_t>(m_GL_max_elements_vertices / m_verticesCount), lines_count);

	m_dbo.bind();

	m_dbo.allocate(lines_count * sizeof(DrawArraysIndirectCommand));
	assert(m_dbo.size() > 0);
	DrawArraysIndirectCommand* dbo_bytes =
	    static_cast<DrawArraysIndirectCommand*>(m_dbo.map(QOpenGLBuffer::WriteOnly));
	assert(dbo_bytes);
	std::generate(dbo_bytes, dbo_bytes + lines_count, [ line = 0, this ]() mutable {
		size_t drawIndex = line++;
		return DrawArraysIndirectCommand{GLuint(m_verticesCount), 1,
		                                 GLuint(drawIndex * m_verticesCount), drawIndex};
	});

	m_dbo.unmap();
	m_dbo.release();

	m_vbo.bind();
	m_vbo.allocate(m_linesPerVboCount * m_verticesCount * sizeof(Vertex));
	assert(m_vbo.size() > 0);
	m_vbo.release();

	m_cbo.bind();
	m_cbo.allocate(m_linesPerVboCount * sizeof(CboBlock));
	assert(m_cbo.size() > 0);
	m_cbo.release();
}

void PVSeriesView::fill_vbo_GL(size_t const lineBegin, size_t const lineEnd)
{
	size_t const line_byte_size = m_verticesCount * sizeof(Vertex);

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

void PVSeriesView::fill_cbo_GL(size_t const lineBegin, size_t const lineEnd)
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

void PVSeriesView::softPaintGL()
{
	// glViewport(0, 0, m_w, m_h);
	// glClear(GL_COLOR_BUFFER_BIT);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, m_fbo->handle());
	// glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebufferObject());
	glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
	                       m_fbtexture->textureId(), 0);
	glBlitFramebuffer(0, 0, m_w, m_h, 0, 0, m_w, m_h, GL_COLOR_BUFFER_BIT, GL_LINEAR);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
	qDebug() << "softPaintGL";
}

void PVSeriesView::onAboutToCompose()
{
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end - startCompositionTimer;
	qDebug() << "compositionGL interval:" << diff.count();
	startCompositionTimer = std::chrono::high_resolution_clock::now();
}

void PVSeriesView::onFrameSwapped()
{
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end - startCompositionTimer;
	qDebug() << "compositionGL:" << diff.count();
}

} // namespace PVParallelView
