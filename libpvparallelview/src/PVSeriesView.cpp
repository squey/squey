#include <pvparallelview/PVSeriesView.h>

#include <cassert>
#include <cstring>
#include <cmath>
#include <mutex>

namespace PVParallelView
{

#define SHADER(x) #x
#define LOAD_GL_FUNC(x)                                                                            \
	x = reinterpret_cast<decltype(x)>(context()->getProcAddress(#x));                              \
	assert(x != nullptr);

struct DrawArraysIndirectCommand {
	GLuint count;
	GLuint instanceCount;
	GLuint first;
	GLuint baseInstance;
};

PVSeriesView::PVSeriesView(Inendi::PVRangeSubSampler& rss, QWidget* parent)
    : PVOpenGLWidget(/*QGLFormat(QGL::SampleBuffers),*/ parent)
    , m_rss(rss)
    , m_dbo(static_cast<QOpenGLBuffer::Type>(GL_DRAW_INDIRECT_BUFFER))
{
}

PVSeriesView::~PVSeriesView()
{
	makeCurrent();
	m_program.release();
	m_dbo.destroy();
	m_vbo.destroy();
	m_vao.destroy();
	doneCurrent();
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

void PVSeriesView::showSeries(std::vector<size_t> seriesDrawOrder)
{
	std::swap(m_seriesDrawOrder, seriesDrawOrder);

	makeCurrent();

	GLint max_elemv = 0;
	glGetIntegerv(GL_MAX_ELEMENTS_VERTICES, &max_elemv);
	m_linesPerVboCount =
	    std::min(static_cast<size_t>(max_elemv / m_verticesCount), m_seriesDrawOrder.size());

	compute_dbo_GL();

	m_vbo.bind();
	m_vbo.allocate(m_linesPerVboCount * m_verticesCount * sizeof(Vertex));
	assert(m_vbo.size() > 0);
	m_vbo.release();

	doneCurrent();
}

void PVSeriesView::initializeGL()
{
	initializeOpenGLFunctions();
	LOAD_GL_FUNC(glMultiDrawArraysIndirect);
	LOAD_GL_FUNC(glMapBufferRange);

	qDebug() << "Context:" << QString(reinterpret_cast<const char*>(glGetString(GL_VERSION)));

	auto start = std::chrono::system_clock::now();

	// glClearColor(1.0, 0.2, 0.8, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);

	m_program = std::make_unique<QOpenGLShaderProgram>(context());

	// clang-format off
	m_program->addShaderFromSourceCode(QOpenGLShader::Vertex,
"#version 450\n" SHADER(
in vec4 vertex;
out vec4 lineColor;
uniform vec4 size;
void main(void) {
    float line = floor(gl_VertexID / size.w);
    lineColor = vec4(fract((line + 1) * 0.7), fract(2 * (line + 1) * 0.7), fract(3 * (line + 1) * 0.7), 1);
    vec4 wvertex = vertex;
    wvertex.y = wvertex.x / 16384;// ((1 << 14) - 1);
    wvertex.x = mod(gl_VertexID, size.w) / (size.w - 1);
    gl_Position.xy = vec2(fma(wvertex.x, 2, -1), fma(wvertex.y, 2, -1));
}));
	m_program->addShaderFromSourceCode(QOpenGLShader::Fragment,
"#version 450\n" SHADER(
in vec4 lineColor; out vec4 FragColor;
void main(void) {
	FragColor = lineColor;
	// FragColor = vec4(1,1,1,1);
}));
	// clang-format on

	m_program->link();
	qDebug() << m_program->log();
	m_program->bind();

	m_vao.create();
	m_vbo.create();
	m_dbo.create();

	m_vao.bind();
	m_program->bind();

	m_sizeLocation = m_program->uniformLocation("size");

	m_vbo.bind();
	m_vbo.setUsagePattern(QOpenGLBuffer::StreamDraw);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 1, GL_UNSIGNED_SHORT, GL_FALSE, 0, 0x0);

	m_vbo.release();

	m_vao.release();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	qDebug() << "initialiseGL:" << diff.count();
}

void PVSeriesView::resizeGL(int w, int h)
{
	qDebug() << "resizeGL(" << w << ", " << h << ")";
	m_vao.bind();
	m_w = w;
	m_h = h;
	glViewport(0, 0, w, h);
	// glFrustum(0,0,0,0,0.5,1);

	m_rss.set_sampling_count(w);
	m_rss.resubsample();
	m_verticesCount = m_rss.samples_count();
	m_linesCount = m_rss.timeseries_count();

	GLint max_elemv = 0;
	glGetIntegerv(GL_MAX_ELEMENTS_VERTICES, &max_elemv);
	m_linesPerVboCount = std::min(static_cast<size_t>(max_elemv / m_verticesCount), m_linesCount);

	m_program->setUniformValue(m_sizeLocation, QVector4D(0, 0, 0, m_verticesCount));

	compute_dbo_GL();

	m_vbo.bind();
	m_vbo.allocate(m_linesPerVboCount * m_verticesCount * sizeof(Vertex));
	assert(m_vbo.size() > 0);
	m_vbo.release();

	m_vao.release();
}

void PVSeriesView::paintGL()
{
	glViewport(0, 0, m_w, m_h);
	auto start = std::chrono::system_clock::now();

	m_vao.bind();

	if (m_backgroundColor) {
		glClearColor(m_backgroundColor->redF(), m_backgroundColor->greenF(),
		             m_backgroundColor->blueF(), m_backgroundColor->alphaF());
		m_backgroundColor = std::nullopt;
	}

	glClear(GL_COLOR_BUFFER_BIT);

	// glEnable(GL_LINE_SMOOTH);
	// glLineWidth(3.f);

	m_dbo.bind();

	m_vbo.bind();

	int vbo_size = m_vbo.size();
	size_t line_byte_size = m_verticesCount * sizeof(Vertex);

	for (size_t line = 0; line < m_seriesDrawOrder.size();) {
		{
			void* vbo_bytes = glMapBufferRange(GL_ARRAY_BUFFER, 0, vbo_size,
			                                   GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT |
			                                       GL_MAP_UNSYNCHRONIZED_BIT);

			assert(vbo_bytes);
			for (size_t line_end = std::min(line + m_linesPerVboCount, m_seriesDrawOrder.size());
			     line < line_end; ++line) {
				// qDebug() << "Line " << line;
				// for(int i = 0; i < m_verticesCount; ++i){
				// 	qDebug() << m_rss.averaged_timeserie(line)[i];
				// }
				std::memcpy(reinterpret_cast<uint8_t*>(vbo_bytes) + line * line_byte_size,
				            reinterpret_cast<uint8_t const*>(
				                m_rss.averaged_timeserie(m_seriesDrawOrder[line]).data()),
				            line_byte_size);
			}
			m_vbo.unmap();
		}

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

	m_vbo.release();

	m_dbo.release();

	// m_vbo.release();
	// m_program->release();
	m_vao.release();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	qDebug() << "paintGL:" << diff.count();

	// debugAvailableMemory();
}

void PVSeriesView::compute_dbo_GL()
{
	m_dbo.bind();

	auto lines_count = m_seriesDrawOrder.size();

	m_dbo.allocate(lines_count * sizeof(DrawArraysIndirectCommand));
	assert(m_dbo.size() > 0);
	DrawArraysIndirectCommand* dbo_bytes =
	    static_cast<DrawArraysIndirectCommand*>(m_dbo.map(QOpenGLBuffer::WriteOnly));
	assert(dbo_bytes);
	std::generate(dbo_bytes, dbo_bytes + lines_count, [ line = 0, this ]() mutable {
		return DrawArraysIndirectCommand{GLuint(m_verticesCount), 1,
		                                 GLuint((line++) * m_verticesCount), 0};
	});

	m_dbo.unmap();
	m_dbo.release();
}

} // namespace PVParallelView
