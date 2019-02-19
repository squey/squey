#include <pvparallelview/PVSeriesRendererOpenGL.h>

#include <cassert>

#include <QResizeEvent>
#include <QSurface>

namespace PVParallelView
{

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

	setDrawMode(PVSeriesView::DrawMode::Lines);
}

PVSeriesRendererOpenGL::~PVSeriesRendererOpenGL()
{
	cleanupGL();
}

bool PVSeriesRendererOpenGL::capability()
{
	static const bool s_opengl_capable = [] {
		QOpenGLContext qogl;
		QSurfaceFormat format;
		format.setVersion(4, 3);
		format.setProfile(QSurfaceFormat::CoreProfile);
		qogl.setFormat(format);
		return qogl.create() && qogl.format().version() >= qMakePair(4, 3);
	}();
	return s_opengl_capable;
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

void PVSeriesRendererOpenGL::setDrawMode(PVSeriesView::DrawMode mode)
{
	m_drawMode = mode;
}

void PVSeriesRendererOpenGL::onShowSeries()
{
	m_needReallocateBuffers = true;
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

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

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
	m_programLines.reset();
	m_programPoints.reset();
	m_dbo.destroy();
	m_cbo.destroy();
	m_vbo.destroy();
	m_vao.destroy();
	doneCurrent();
	m_wasCleanedUp = true;
}

void PVSeriesRendererOpenGL::resizeEvent(QResizeEvent* event)
{
	if (not m_wasCleanedUp and event->size() == m_oldSize) {
		return;
	}
	m_blockPaint = true;
	QOpenGLWidget::resizeEvent(event);
	m_blockPaint = false;
	m_oldSize = size();
}

void PVSeriesRendererOpenGL::resizeGL(int w, int h)
{
	qDebug() << "resizeGL(" << w << ", " << h << ")";
	m_needReallocateBuffers = true;
}

void PVSeriesRendererOpenGL::paintGL()
{
	if (m_blockPaint or not m_rss.valid()) {
		return;
	}

	glViewport(0, 0, width(), height());
	auto start = std::chrono::system_clock::now();

	m_vao.bind();

	if (m_backgroundColor) {
		glClearColor(m_backgroundColor->redF(), m_backgroundColor->greenF(),
		             m_backgroundColor->blueF(), m_backgroundColor->alphaF());
		m_backgroundColor = std::nullopt;
	}

	if (m_drawMode) {
		setDrawMode_GL(*m_drawMode);
		m_drawMode = std::nullopt;
	}

	m_program->setUniformValue(m_sizeLocation, QVector4D(0, 0, m_rss.samples_count(), width()));

	glClear(GL_COLOR_BUFFER_BIT);

	if (m_needReallocateBuffers) {
		m_needReallocateBuffers = false;
		compute_dbo_GL();
	}

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

void PVSeriesRendererOpenGL::setDrawMode_GL(PVSeriesView::DrawMode mode)
{
	if (mode == PVSeriesView::DrawMode::Lines) {
		// glEnable(GL_LINE_SMOOTH);
		// glLineWidth(3.f);
		m_program = m_programLines.get();
		m_glDrawMode = GL_LINE_STRIP;
	} else if (mode == PVSeriesView::DrawMode::Points) {
		// glPointSize(5.f);
		m_program = m_programPoints.get();
		m_glDrawMode = GL_POINTS;
	}
	m_program->bind();
	m_sizeLocation = m_program->uniformLocation("size");
}

void PVSeriesRendererOpenGL::compute_dbo_GL()
{
	auto lines_count = m_seriesDrawOrder.size();

	if (lines_count <= 0 or m_rss.samples_count() <= 0) {
		return;
	}

	m_linesPerVboCount = std::min(
	    static_cast<size_t>(m_GL_max_elements_vertices / m_rss.samples_count()), lines_count);

	fill_dbo_GL();

	m_vbo.bind();
	m_vbo.allocate(m_linesPerVboCount * m_rss.samples_count() * sizeof(Vertex));
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
	std::generate(dbo_bytes, dbo_bytes + m_linesPerVboCount, [ line = 0u, this ]() mutable {
		uint32_t drawIndex = line++;
		return DrawArraysIndirectCommand{GLuint(m_rss.samples_count()), 1,
		                                 GLuint(drawIndex * m_rss.samples_count()), drawIndex};
	});

	m_dbo.unmap();
	m_dbo.release();
}

void PVSeriesRendererOpenGL::fill_vbo_GL(size_t const lineBegin, size_t const lineEnd)
{
	size_t const line_byte_size = m_rss.samples_count() * sizeof(Vertex);

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
	glMultiDrawArraysIndirect(m_glDrawMode, 0, lineEnd - lineBegin,
	                          sizeof(DrawArraysIndirectCommand));
	// glMultiDrawArraysIndirect(GL_POINTS, 0, lineEnd - lineBegin,
	//                           sizeof(DrawArraysIndirectCommand));
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
	std::string_view vertexShader =
"#version 430\n" SHADER(
in vec4 vertex;
in vec3 color;
smooth out vec4 lineColor;
uniform vec4 size;
void main(void) {
    //float line = floor(gl_VertexID / size.w);
    //float line = gl_InstanceID;
    //lineColor = vec4(fract((line + 1) * 0.7), fract(2 * (line + 1) * 0.7), fract(3 * (line + 1) * 0.7), 1);
    lineColor = vec4(color, 1);
    //lineColor = vec4(size.rgb, 1);
    vec4 wvertex = vertex;
    wvertex.y = vertex.x / ((1 << 14) - 1);
    wvertex.x = (gl_VertexID % int(size.z)) / (size.z - 1);
    int vx = int(vertex.x);
    if(bool(vx & (1 << 15))) { //if out of range
    	if(bool(vx & (1 << 14))) { //if overflow
    		wvertex.y = 1.5;
    	} else { //else underflow
    		wvertex.y = -0.5;
    	}
    } else if(bool(vx & (1 << 14))) { //else if no value
    	//lineColor = vec4(0.,0.,0.,0.);
    	wvertex.y = 2.5;
    	wvertex.z = 1;
    }
    gl_Position.xyz = vec3(fma(wvertex.x, 2.0, -1.0), fma(wvertex.y, 2.0, -1.0), wvertex.z);
});

	std::string_view geometryShaderLines =
"#version 430\n" SHADER(
layout(lines) in;
smooth in vec4 lineColor[];
layout(line_strip, max_vertices = 2) out;
smooth out vec4 geolineColor;

uniform vec4 size;

void main(void) {
	if (gl_in[0].gl_Position.y > 2) {
		return;
	}
	geolineColor = lineColor[0];
	gl_Position = gl_in[0].gl_Position;
	EmitVertex();
	if (gl_in[1].gl_Position.y > 2) {
		geolineColor = lineColor[0];
		bool adjust = gl_in[0].gl_Position.x < 0;
		gl_Position = gl_in[0].gl_Position + vec4(fma(float(adjust), 2, -1)*2/size.w, 0, 0, 0);
		EmitVertex();
		return;
	}
	geolineColor = lineColor[1];
	gl_Position = gl_in[1].gl_Position;
	EmitVertex();
});

	std::string_view fragmentShaderLines =
"#version 430\n" SHADER(
smooth in vec4 geolineColor;
out vec4 FragColor;
void main(void) {
	FragColor = geolineColor;
});

	std::string_view fragmentShaderPoints =
"#version 430\n" SHADER(
smooth in vec4 lineColor;
out vec4 FragColor;
void main(void) {
	FragColor = lineColor;
	//FragColor = vec4(lineColor.rgb, 1 - smoothstep(0.3, 0.5, distance(vec2(0.5, 0.5), gl_PointCoord)));
});

	m_programLines = std::make_unique<QOpenGLShaderProgram>(context());
	m_programLines->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShader.data());
	m_programLines->addShaderFromSourceCode(QOpenGLShader::Geometry, geometryShaderLines.data());
	m_programLines->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderLines.data());
	m_programLines->link();
	qDebug() << m_programLines->log();

	m_programPoints = std::make_unique<QOpenGLShaderProgram>(context());
	m_programPoints->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShader.data());
	m_programPoints->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderPoints.data());
	m_programPoints->link();
	qDebug() << m_programPoints->log();
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
