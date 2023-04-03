//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvparallelview/PVSeriesRendererOpenGL.h>

#include <pvkernel/core/PVConfig.h>
#include <QSettings>

#include <cassert>

#include <QResizeEvent>
#include <QSurface>
#include <EGL/egl.h>

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
    , _dbo(static_cast<QOpenGLBuffer::Type>(GL_DRAW_INDIRECT_BUFFER))
{
	QSurfaceFormat format;
	format.setRenderableType(QSurfaceFormat::OpenGLES);
	format.setVersion(OpenGLES_version_major, OpenGLES_version_minor);
	format.setProfile(QSurfaceFormat::CoreProfile);
	setFormat(format);

	// setUpdateBehavior(QOpenGLWidget::PartialUpdate);

	set_draw_mode(PVSeriesView::DrawMode::Lines);
}

PVSeriesRendererOpenGL::~PVSeriesRendererOpenGL() noexcept
{
	cleanupGL();
}

bool PVSeriesRendererOpenGL::capability()
{
	static const bool s_opengl_capable = [] {
		if (PVCore::PVConfig::get().config().value("backend_opencl/force_cpu", false).toBool()) {
			qDebug() << "backend_opencl/force_cpu is set to true in user config";
			return false;
		}
		QOpenGLWidget qoglwg;
		QSurfaceFormat format;
		format.setRenderableType(QSurfaceFormat::OpenGLES);
		format.setVersion(OpenGLES_version_major, OpenGLES_version_minor);
		format.setProfile(QSurfaceFormat::CoreProfile);
		qoglwg.setFormat(format);
		qoglwg.resize(20, 20);
		if (qoglwg.grabFramebuffer().isNull()) {
			qDebug() << "Could not use a QOpenGLWidget to grab framebuffer";
		} else if (not qoglwg.isValid()) {
			qDebug() << "Could not create a valid QOpenGLWidget";
		} else if (qoglwg.format().version() <
		           qMakePair(OpenGLES_version_major, OpenGLES_version_minor)) {
			qDebug() << "Expecting" << qMakePair(OpenGLES_version_major, OpenGLES_version_minor)
			         << "but QOpenGLWidget could only deliver " << qoglwg.format().version();
		} else {
			qoglwg.makeCurrent();
			if (auto vendor = qoglwg.context()->functions()->glGetString(GL_VENDOR); reinterpret_cast<const char*>(vendor) != std::string("NVIDIA")) {
				qDebug() << "Unsupported GL_VENDOR (" << vendor << ") currently supports only NVIDIA";
				return false;
			}
			return true;
		}
		return false;
	}();
	return s_opengl_capable;
}

auto PVSeriesRendererOpenGL::capability(PVSeriesView::DrawMode mode) -> PVSeriesView::DrawMode
{
	if (mode == PVSeriesView::DrawMode::Lines || mode == PVSeriesView::DrawMode::Points ||
	    mode == PVSeriesView::DrawMode::LinesAlways) {
		return mode;
	}
	return PVSeriesView::DrawMode::Lines;
}

void PVSeriesRendererOpenGL::debug_available_memory()
{
#define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049

	GLint total_mem_kb = 0;
	glGetIntegerv(GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX, &total_mem_kb);

	GLint cur_avail_mem_kb = 0;
	glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &cur_avail_mem_kb);

	qDebug() << cur_avail_mem_kb << " available on total " << total_mem_kb;
}

void PVSeriesRendererOpenGL::debug_errors()
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

void PVSeriesRendererOpenGL::set_background_color(QColor const& bgcol)
{
	_background_color = bgcol;
	setPalette(QPalette(bgcol));
}

void PVSeriesRendererOpenGL::set_draw_mode(PVSeriesView::DrawMode mode)
{
	_draw_mode = capability(mode);
	_need_to_reset_draw_mode = true;
}

void PVSeriesRendererOpenGL::initializeGL()
{
	connect(context(), &QOpenGLContext::aboutToBeDestroyed, this,
	        &PVSeriesRendererOpenGL::cleanupGL);

	initializeOpenGLFunctions();
	LOAD_GL_FUNC(glMultiDrawArraysIndirect);

	glGetIntegerv(GL_MAX_ELEMENTS_VERTICES, &_GL_max_elements_vertices);
	qDebug() << "GL_MAX_ELEMENTS_VERTICES" << _GL_max_elements_vertices;
	qDebug() << "Context:" << QString(reinterpret_cast<const char*>(glGetString(GL_VERSION)));
	qDebug() << "Screen:" << context()->screen();
	qDebug() << "Surface:" << context()->surface()->surfaceClass()
	         << context()->surface()->surfaceType()
	         << "(supports OpenGL:" << context()->surface()->supportsOpenGL() << ")"
	         << context()->surface()->size();

	auto start = std::chrono::system_clock::now();

	glClear(GL_COLOR_BUFFER_BIT);

	setup_shaders_GL();

	_vao.create();
	_vbo.create();
	_cbo.create();
	_dbo.create();

	_vao.bind();

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	{
		_vbo.bind();
		_vbo.setUsagePattern(QOpenGLBuffer::StreamDraw);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 1, GL_UNSIGNED_SHORT, GL_FALSE, 0, nullptr);

		_vbo.release();
	}

	{
		_cbo.bind();
		_cbo.setUsagePattern(QOpenGLBuffer::StreamDraw);

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
		glVertexAttribDivisor(1, 1);

		_cbo.release();
	}

	_vao.release();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	qDebug() << "initialiseGL:" << diff.count();

	debug_errors();

	if (_was_cleaned_up) {
		_was_cleaned_up = false;
		resizeGL(width(), height());
		update();
	}
}

void PVSeriesRendererOpenGL::cleanupGL()
{
	if (_was_cleaned_up) {
		return;
	}
	qDebug() << "cleanupGL";
	makeCurrent();
	_program_Lines.reset();
	_program_Points.reset();
	_dbo.destroy();
	_cbo.destroy();
	_vbo.destroy();
	_vao.destroy();
	doneCurrent();
	_was_cleaned_up = true;
}

void PVSeriesRendererOpenGL::resizeEvent(QResizeEvent* event)
{
	if (not _was_cleaned_up and event->size() == _old_size) {
		return;
	}
	_block_paint = true;
	QOpenGLWidget::resizeEvent(event);
	_block_paint = false;
	_old_size = size();
}

void PVSeriesRendererOpenGL::resizeGL(int w, int h)
{
	qDebug() << "resizeGL(" << w << ", " << h << ")";
}

void PVSeriesRendererOpenGL::paintGL()
{
	if (_block_paint or not _rss.valid() or _rss.samples_count() <= 0 or
	    _series_draw_order.empty()) {
		if (_background_color) {
			glClearColor(_background_color->redF(), _background_color->greenF(),
			             _background_color->blueF(), _background_color->alphaF());
		}
		glClear(GL_COLOR_BUFFER_BIT);
		return;
	}

	glViewport(0, 0, width(), height());
	auto start = std::chrono::system_clock::now();

	_vao.bind();

	if (_background_color) {
		glClearColor(_background_color->redF(), _background_color->greenF(),
		             _background_color->blueF(), _background_color->alphaF());
		_background_color = std::nullopt;
	}

	if (_need_to_reset_draw_mode) {
		_need_to_reset_draw_mode = false;
		set_draw_mode_GL();
	}

	_program->setUniformValue(_size_location, QVector4D(0, 0, _rss.samples_count(), width()));

	glClear(GL_COLOR_BUFFER_BIT);

	_lines_per_vbo_count = lines_per_vbo();

	fill_dbo_GL();

	for (size_t line = 0; line < _series_draw_order.size();) {
		auto line_end = std::min(line + _lines_per_vbo_count, _series_draw_order.size());
		fill_vbo_GL(line, line_end);
		fill_cbo_GL(line, line_end);
		draw_GL(line, line_end);
		line = line_end;
	}

	_vao.release();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	qDebug() << "paintGL:" << diff.count();

	debug_errors();
}

void PVSeriesRendererOpenGL::set_draw_mode_GL()
{
	if (_draw_mode == PVSeriesView::DrawMode::Lines) {
		// glEnable(GL_LINE_SMOOTH);
		// glLineWidth(3.f);
		_program = _program_Lines.get();
		_gl_draw_mode = GL_LINE_STRIP;
	} else if (_draw_mode == PVSeriesView::DrawMode::Points) {
		// glPointSize(5.f);
		_program = _program_Points.get();
		_gl_draw_mode = GL_POINTS;
	} else if (_draw_mode == PVSeriesView::DrawMode::LinesAlways) {
		_program = _program_LinesAlways.get();
		_gl_draw_mode = GL_LINE_STRIP_ADJACENCY;
	}
	_program->bind();
	_size_location = _program->uniformLocation("size");
}

int PVSeriesRendererOpenGL::lines_per_vbo() const
{
	size_t vertices_per_line = _rss.samples_count();
	return std::min(static_cast<size_t>(_GL_max_elements_vertices / vertices_per_line),
	                _series_draw_order.size());
}

void PVSeriesRendererOpenGL::allocate_buffer_GL(QOpenGLBuffer& buffer, int expected_size)
{
	if (buffer.size() != expected_size) {
		buffer.allocate(expected_size);
		assert(buffer.size() == expected_size);
	}
}

void PVSeriesRendererOpenGL::fill_dbo_GL()
{
	size_t vertices_per_line = _rss.samples_count();
	_dbo.bind();
	allocate_buffer_GL(_dbo, _lines_per_vbo_count * sizeof(DrawArraysIndirectCommand));
	auto* dbo_bytes =
	    static_cast<DrawArraysIndirectCommand*>(_dbo.map(QOpenGLBuffer::WriteOnly));
	assert(dbo_bytes);
	std::generate(dbo_bytes, dbo_bytes + _lines_per_vbo_count,
	              [line = 0u, vertices_per_line]() mutable {
		              uint32_t draw_index = line++;
		              return DrawArraysIndirectCommand{GLuint(vertices_per_line), 1,
		                                               GLuint(draw_index * vertices_per_line),
		                                               draw_index};
	              });
	_dbo.unmap();
	_dbo.release();
}

void PVSeriesRendererOpenGL::fill_vbo_GL(size_t const line_begin, size_t const line_end)
{
	size_t vertices_per_line = _rss.samples_count();
	size_t const line_byte_size = vertices_per_line * sizeof(Vertex);

	_vbo.bind();
	allocate_buffer_GL(_vbo, _lines_per_vbo_count * line_byte_size);
	void* vbo_bytes = glMapBufferRange(GL_ARRAY_BUFFER, 0, _vbo.size(),
	                                   GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT |
	                                       GL_MAP_UNSYNCHRONIZED_BIT);
	assert(vbo_bytes);

	if (_draw_mode == PVSeriesView::DrawMode::LinesAlways) {
		auto next_valid = [](auto begin, auto end) {
			return std::find_if(
			    begin, end, [](auto x) { return not PVRSS::display_match(x, PVRSS::no_value); });
		};
#pragma omp parallel for
		for (size_t line = line_begin; line < line_end; ++line) {
			auto const& av_ts = _rss.sampled_timeserie(_series_draw_order[line].dataIndex);
			Vertex* vertex_bytes =
			    reinterpret_cast<Vertex*>(vbo_bytes) + (line - line_begin) * vertices_per_line;
			for (size_t j = 0; j < vertices_per_line; ++j) {
				auto vertex = av_ts[j];
				if (PVRSS::display_match(vertex, PVRSS::no_value)) {
					if (j < 1 or not PVRSS::display_match(av_ts[j - 1], PVRSS::no_value)) {
						auto next_valid_it = next_valid(av_ts.begin() + j, av_ts.end());
						vertex_bytes[j] = Vertex{GLushort(
						    PVRSS::no_value + std::distance(av_ts.begin() + j, next_valid_it))};
						continue;
					} else if (j < 2 or not PVRSS::display_match(av_ts[j - 2], PVRSS::no_value)) {
						auto next_valid_it = next_valid(av_ts.begin() + j, av_ts.end());
						auto next_valid_value =
						    next_valid_it != av_ts.end() ? *next_valid_it : av_ts[j - 2];
						if (PVRSS::display_match(next_valid_value, PVRSS::overflow_value)) {
							next_valid_value = PVRSS::display_type_max_val;
						} else if (PVRSS::display_match(next_valid_value, PVRSS::underflow_value)) {
							next_valid_value = PVRSS::display_type_min_val;
						}
						vertex_bytes[j] = Vertex{GLushort(PVRSS::no_value + next_valid_value)};
						continue;
					}
				}
				vertex_bytes[j] = Vertex{vertex};
			}
		}
	} else {
		for (size_t line = line_begin; line < line_end; ++line) {
			std::memcpy(reinterpret_cast<uint8_t*>(vbo_bytes) +
			                (line - line_begin) * line_byte_size,
			            reinterpret_cast<uint8_t const*>(
			                _rss.sampled_timeserie(_series_draw_order[line].dataIndex).data()),
			            line_byte_size);
		}
	}

	_vbo.unmap();
	_vbo.release();
}

void PVSeriesRendererOpenGL::fill_cbo_GL(size_t const line_begin, size_t const line_end)
{
	_cbo.bind();
	allocate_buffer_GL(_cbo, _lines_per_vbo_count * sizeof(CboBlock));
	auto* cbo_bytes = static_cast<CboBlock*>(glMapBufferRange(
	    GL_ARRAY_BUFFER, 0, _cbo.size(),
	    GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT));
	assert(cbo_bytes);
	for (size_t line = line_begin; line < line_end; ++line) {
		cbo_bytes[line - line_begin].colorR = _series_draw_order[line].color.redF();
		cbo_bytes[line - line_begin].colorG = _series_draw_order[line].color.greenF();
		cbo_bytes[line - line_begin].colorB = _series_draw_order[line].color.blueF();
	}
	_cbo.unmap();
	_cbo.release();
}

void PVSeriesRendererOpenGL::draw_GL(size_t const line_begin, size_t const line_end)
{
	_dbo.bind();
	glMultiDrawArraysIndirect(_gl_draw_mode, nullptr, line_end - line_begin,
	                          sizeof(DrawArraysIndirectCommand));
	_dbo.release();

	// for (size_t line = 0; line < line_end - line_begin; ++line) {
	// 	_program->setUniformValue(_size_location, QVector4D(_series_draw_order[line].color.redF(),
	// 	                                                     _series_draw_order[line].color.greenF(),
	// 	                                                     _series_draw_order[line].color.blueF(),
	// 	                                                     width()));
	// 	glDrawArrays(GL_LINE_STRIP, line * width(), width());
	// }
}

// clang-format off
void PVSeriesRendererOpenGL::setup_shaders_GL()
{
	std::string_view vertex_shader =
"#version 320 es\n" R"(
in vec4 vertex;
in vec3 color;
smooth out vec4 lineColor;
uniform vec4 size;
const int max_value = (1 << 14) - 1;
void main(void) {
	lineColor = vec4(color, 1);
	//lineColor = vec4(size.rgb, 1);
	vec4 wvertex = vertex;
	wvertex.y = vertex.x / float(max_value);
	wvertex.x = float(gl_VertexID % int(size.z)) / (size.z - 1.0);
	wvertex.xy = vec2(fma(wvertex.x, 2.0, -1.0), fma(wvertex.y, 2.0, -1.0));
	int vx = int(vertex.x);
	if(bool(vx & (1 << 15))) { //if out of range
		if(bool(vx & (1 << 14))) { //if overflow
			wvertex.y = 1.001;
		} else { //else underflow
			wvertex.y = -1.001;
		}
	} else if(bool(vx & (1 << 14))) { //else if no value
		lineColor = vec4(color, 0);
		wvertex.xy = vec2(gl_VertexID, vertex.x);
	}
	gl_Position.xyzw = wvertex.xyzw;
})";

	std::string_view geometry_shader_Lines =
"#version 320 es\n" R"(
layout(lines) in;
smooth in vec4 lineColor[];
layout(line_strip, max_vertices = 2) out;
smooth out vec4 geolineColor;

uniform vec4 size;

void main(void) {
	if (lineColor[0].a == 0.0) {
		return;
	}
	geolineColor = lineColor[0];
	gl_Position = gl_in[0].gl_Position;
	EmitVertex();
	if (lineColor[1].a == 0.0) {
		geolineColor = lineColor[0];
		bool adjust = gl_in[0].gl_Position.x < 0.0;
		gl_Position.xy = gl_in[0].gl_Position.xy + vec2(fma(float(adjust), 2.0, -1.0)*2.0/size.w, 0.0);
		EmitVertex();
		return;
	}
	geolineColor = lineColor[0];
	gl_Position = gl_in[1].gl_Position;
	EmitVertex();
})";

	std::string_view geometry_shader_LinesAlways =
"#version 320 es\n" R"(
layout(lines_adjacency) in;
smooth in vec4 lineColor[];
layout(line_strip, max_vertices = 6) out;
smooth out vec4 geolineColor;

uniform vec4 size;

const int max_value = (1 << 14) - 1;

void emitShortVertex(in const int index)
{
	geolineColor = lineColor[index];
	gl_Position.xy = gl_in[index].gl_Position.xy;
	EmitVertex();
}

vec2 emitLongVertex(in const int index)
{
	geolineColor = vec4(lineColor[index].rgb, 1.0);
	int vertexId = int(gl_in[index].gl_Position.x) % int(size.z) + (int(gl_in[index].gl_Position.y) & max_value);
	int valindex = index + 1;
	vec2 wvertex;
	wvertex.x = float(vertexId) / (size.z - 1.0);
	wvertex.x = fma(wvertex.x, 2.0, -1.0);
	if (lineColor[valindex].a == 0.0) {
		wvertex.y = float(int(gl_in[valindex].gl_Position.y) & max_value) / float(max_value);
		wvertex.y = fma(wvertex.y, 2.0, -1.0);
	} else {
		wvertex.y = gl_in[valindex].gl_Position.y;
	}
	gl_Position.xy = wvertex;
	EmitVertex();
	return wvertex;
}

void main(void) {
	if (gl_PrimitiveIDIn == 0) {
		if (lineColor[0].a != 0.0 && lineColor[1].a != 0.0) {
			emitShortVertex(0);
			emitShortVertex(1);
		} else if (lineColor[0].a != 0.0) {
			emitShortVertex(0);
			emitLongVertex(1);
		} else {
			vec2 wert = emitLongVertex(0);
			geolineColor = vec4(lineColor[0].rgb, 1.0);
			gl_Position.xy = vec2(-1.0, wert.y);
			EmitVertex();
		}
		EndPrimitive();
	}
	if (gl_PrimitiveIDIn == int(size.z) - 4 && lineColor[2].a != 0.0 && lineColor[3].a != 0.0) {
		emitShortVertex(2);
		emitShortVertex(3);
		EndPrimitive();
	}
	if (lineColor[1].a == 0.0) {
		return;
	}
	emitShortVertex(1);
	if (lineColor[2].a == 0.0) {
		emitLongVertex(2);
	} else {
		emitShortVertex(2);
	}
})";

	std::string_view fragment_shader_Lines =
"#version 320 es\n" R"(
smooth in mediump vec4 geolineColor;
out mediump vec4 FragColor;
void main(void) {
	FragColor = geolineColor;
})";

	std::string_view fragment_shader_Points =
"#version 320 es\n" R"(
smooth in mediump vec4 lineColor;
out mediump vec4 FragColor;
void main(void) {
	FragColor = lineColor;
})";

	_program_Lines = std::make_unique<QOpenGLShaderProgram>(context());
	_program_Lines->addShaderFromSourceCode(QOpenGLShader::Vertex, vertex_shader.data());
	_program_Lines->addShaderFromSourceCode(QOpenGLShader::Geometry, geometry_shader_Lines.data());
	_program_Lines->addShaderFromSourceCode(QOpenGLShader::Fragment, fragment_shader_Lines.data());
	_program_Lines->link();
	qDebug() << _program_Lines->log();

	_program_Points = std::make_unique<QOpenGLShaderProgram>(context());
	_program_Points->addShaderFromSourceCode(QOpenGLShader::Vertex, vertex_shader.data());
	_program_Points->addShaderFromSourceCode(QOpenGLShader::Fragment, fragment_shader_Points.data());
	_program_Points->link();
	qDebug() << _program_Points->log();

	_program_LinesAlways = std::make_unique<QOpenGLShaderProgram>(context());
	_program_LinesAlways->addShaderFromSourceCode(QOpenGLShader::Vertex, vertex_shader.data());
	_program_LinesAlways->addShaderFromSourceCode(QOpenGLShader::Geometry, geometry_shader_LinesAlways.data());
	_program_LinesAlways->addShaderFromSourceCode(QOpenGLShader::Fragment, fragment_shader_Lines.data());
	_program_LinesAlways->link();
	qDebug() << _program_LinesAlways->log();
}
// clang-format on

} // namespace PVParallelView
