//
// MIT License
//
// © Squey, 2026
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

#include <pvparallelview/PVSeriesRendererQRhi.h>

#include <pvparallelview/PVSeriesSampleRows.h>

#include <QFile>
#include <QGuiApplication>
#include <QLoggingCategory>

#include <rhi/qrhi.h>

#if QT_CONFIG(vulkan)
#include <QVulkanInstance>
#endif

#include <algorithm>
#include <memory>

namespace PVParallelView
{

namespace
{

QShader load_shader(const char* path)
{
	QFile file(QString::fromLatin1(path));
	if (not file.open(QIODevice::ReadOnly)) {
		return {};
	}
	return QShader::fromSerialized(file.readAll());
}

// What the shader reads out of its uniform block, laid out to match it, padded to the
// sixteen bytes std140 rounds a block up to.
struct ShaderParams {
	qint32 samples_count;
	qint32 width;
	qint32 height;
	qint32 points_mode;
	float y_sign;
	float padding[3];
};

} // namespace

/**
 * Everything QRhi hands out has to be destroyed before the QRhi itself, and in reverse
 * order of creation. Keeping it all in one place, behind unique_ptrs declared in creation
 * order, is what makes the teardown a matter of letting the object go.
 */
class PVSeriesRhiContext
{
  public:
	static std::unique_ptr<PVSeriesRhiContext> create()
	{
		auto context = std::unique_ptr<PVSeriesRhiContext>(new PVSeriesRhiContext);
		if (not context->init()) {
			return {};
		}
		return context;
	}

	~PVSeriesRhiContext()
	{
		// Declared order is creation order, so releasing back to front is enough.
		_pipeline.reset();
		_bindings.reset();
		_sampler.reset();
		_colors_texture.reset();
		_rows_texture.reset();
		_uniform_buffer.reset();
		_render_target.reset();
		_render_pass.reset();
		_colour_texture.reset();
		_rhi.reset();
	}

	QRhi* rhi() const { return _rhi.get(); }

	/**
	 * Draws the rows of every serie and reads the picture back. Returns a null image if
	 * anything on the way refuses, so the view can fall back rather than show a hole.
	 */
	QImage render(QSize const& size,
	              std::vector<int16_t> const& rows,
	              std::vector<PVSeriesView::SerieDrawInfo> const& draw_order,
	              int samples_count,
	              QColor const& background,
	              PVSeriesView::DrawMode draw_mode)
	{
		const int series_count = int(draw_order.size());
		const bool points_mode = draw_mode == PVSeriesView::DrawMode::Points;

		if (not resize(size) or not upload_capacity(samples_count, series_count)) {
			return {};
		}
		if (not ensure_pipeline()) {
			return {};
		}

		QRhiResourceUpdateBatch* updates = _rhi->nextResourceUpdateBatch();

		// The row of a sample, offset by one so that zero means "no value", spread over
		// the two channels of an RG8 texel: no integer texture format is guaranteed
		// everywhere, and this costs one shift in the shader.
		_rows_staging.resize(size_t(samples_count) * series_count * 2);
		for (int s = 0; s < series_count; ++s) {
			const int16_t* serie_rows = rows.data() + size_t(s) * samples_count;
			uchar* out = _rows_staging.data() + size_t(s) * samples_count * 2;
			for (int j = 0; j < samples_count; ++j) {
				const int encoded =
				    serie_rows[j] == PVSeriesSampleRows::no_row ? 0 : serie_rows[j] + 2;
				out[2 * j] = uchar(encoded & 0xff);
				out[2 * j + 1] = uchar((encoded >> 8) & 0xff);
			}
		}
		QRhiTextureSubresourceUploadDescription rows_upload(_rows_staging.data(),
		                                                    int(_rows_staging.size()));
		rows_upload.setSourceSize(QSize(samples_count, series_count));
		updates->uploadTexture(_rows_texture.get(),
		                       QRhiTextureUploadDescription({0, 0, rows_upload}));

		_colors_staging.resize(size_t(series_count) * 4);
		for (int s = 0; s < series_count; ++s) {
			const QColor& color = draw_order[s].color;
			_colors_staging[4 * s] = uchar(color.red());
			_colors_staging[4 * s + 1] = uchar(color.green());
			_colors_staging[4 * s + 2] = uchar(color.blue());
			_colors_staging[4 * s + 3] = 255;
		}
		QRhiTextureSubresourceUploadDescription colors_upload(_colors_staging.data(),
		                                                      int(_colors_staging.size()));
		colors_upload.setSourceSize(QSize(series_count, 1));
		updates->uploadTexture(_colors_texture.get(),
		                       QRhiTextureUploadDescription({0, 0, colors_upload}));

		const ShaderParams params{samples_count,
		                          size.width(),
		                          size.height(),
		                          points_mode ? 1 : 0,
		                          _rhi->isYUpInNDC() ? -1.0f : 1.0f,
		                          {}};
		updates->updateDynamicBuffer(_uniform_buffer.get(), 0, sizeof(ShaderParams), &params);

		QRhiCommandBuffer* cb = nullptr;
		if (_rhi->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) {
			return {};
		}

		cb->beginPass(_render_target.get(),
		              QColor::fromRgbF(background.redF(), background.greenF(), background.blueF()),
		              {1.0f, 0}, updates);
		cb->setGraphicsPipeline(_pipeline.get());
		cb->setViewport({0, 0, float(size.width()), float(size.height())});
		cb->setShaderResources(_bindings.get());
		cb->draw(quint32(samples_count) * series_count * 2);
		cb->endPass();

		QRhiReadbackResult readback;
		QRhiResourceUpdateBatch* readback_batch = _rhi->nextResourceUpdateBatch();
		readback_batch->readBackTexture({_colour_texture.get()}, &readback);
		cb->resourceUpdate(readback_batch);

		if (_rhi->endOffscreenFrame() != QRhi::FrameOpSuccess or readback.data.isEmpty()) {
			return {};
		}

		QImage image(reinterpret_cast<const uchar*>(readback.data.constData()),
		             readback.pixelSize.width(), readback.pixelSize.height(),
		             QImage::Format_RGBA8888);
		// The readback aliases a QByteArray about to go out of scope, and half of the
		// backends hand the rows back bottom-up.
		return _rhi->isYUpInFramebuffer() ? image.flipped(Qt::Vertical) : image.copy();
	}

  private:
	PVSeriesRhiContext() = default;

	bool init()
	{
		// Every backend reaches the driver through the platform integration, and asking
		// for one without a GUI application does not fail, it walks off a null pointer.
		if (qobject_cast<QGuiApplication*>(QCoreApplication::instance()) == nullptr) {
			return false;
		}

		// Vulkan on Linux, Metal on macOS, Direct3D on Windows: one shader bundle covers
		// all of them, which is the point of going through QRhi at all.
#if defined(Q_OS_MACOS) || defined(Q_OS_IOS)
		QRhiMetalInitParams params;
		_rhi.reset(QRhi::create(QRhi::Metal, &params));
#elif defined(Q_OS_WIN)
		QRhiD3D11InitParams params;
		_rhi.reset(QRhi::create(QRhi::D3D11, &params));
#elif QT_CONFIG(vulkan)
		_vulkan_instance = std::make_unique<QVulkanInstance>();
		// No surface extensions are asked for: this renderer never presents to a window,
		// and requiring them would turn away the software drivers that a machine without
		// a GPU falls back on.
		if (not _vulkan_instance->create()) {
			_vulkan_instance.reset();
			return false;
		}
		QRhiVulkanInitParams params;
		params.inst = _vulkan_instance.get();
		_rhi.reset(QRhi::create(QRhi::Vulkan, &params));
#endif
		if (not _rhi) {
			return false;
		}

		const QShader vertex_shader = load_shader(":/shaders/series.vert.qsb");
		const QShader fragment_shader = load_shader(":/shaders/series.frag.qsb");
		if (not vertex_shader.isValid() or not fragment_shader.isValid()) {
			_rhi.reset();
			return false;
		}
		_vertex_shader = vertex_shader;
		_fragment_shader = fragment_shader;

		_uniform_buffer.reset(_rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
		                                      sizeof(ShaderParams)));
		if (not _uniform_buffer->create()) {
			_rhi.reset();
			return false;
		}
		_sampler.reset(_rhi->newSampler(QRhiSampler::Nearest, QRhiSampler::Nearest,
		                                QRhiSampler::None, QRhiSampler::ClampToEdge,
		                                QRhiSampler::ClampToEdge));
		if (not _sampler->create()) {
			_rhi.reset();
			return false;
		}
		return true;
	}

	bool resize(QSize const& size)
	{
		if (_colour_texture and _colour_texture->pixelSize() == size) {
			return true;
		}
		_render_target.reset();
		_render_pass.reset();
		_colour_texture.reset();

		_colour_texture.reset(_rhi->newTexture(QRhiTexture::RGBA8, size, 1,
		                                       QRhiTexture::RenderTarget |
		                                           QRhiTexture::UsedAsTransferSource));
		if (not _colour_texture->create()) {
			_colour_texture.reset();
			return false;
		}
		_render_target.reset(
		    _rhi->newTextureRenderTarget({{_colour_texture.get()}}));
		_render_pass.reset(_render_target->newCompatibleRenderPassDescriptor());
		_render_target->setRenderPassDescriptor(_render_pass.get());
		if (not _render_target->create()) {
			_render_target.reset();
			_render_pass.reset();
			_colour_texture.reset();
			return false;
		}
		// The pipeline is tied to the render pass descriptor it was built against.
		_pipeline.reset();
		return true;
	}

	bool upload_capacity(int samples_count, int series_count)
	{
		const QSize rows_size(samples_count, series_count);
		if (not _rows_texture or _rows_texture->pixelSize() != rows_size) {
			_rows_texture.reset(_rhi->newTexture(QRhiTexture::RG8, rows_size));
			if (not _rows_texture->create()) {
				_rows_texture.reset();
				return false;
			}
			_bindings.reset();
		}
		const QSize colors_size(series_count, 1);
		if (not _colors_texture or _colors_texture->pixelSize() != colors_size) {
			_colors_texture.reset(_rhi->newTexture(QRhiTexture::RGBA8, colors_size));
			if (not _colors_texture->create()) {
				_colors_texture.reset();
				return false;
			}
			_bindings.reset();
		}
		return true;
	}

	bool ensure_pipeline()
	{
		if (not _bindings) {
			_bindings.reset(_rhi->newShaderResourceBindings());
			_bindings->setBindings(
			    {QRhiShaderResourceBinding::uniformBuffer(0, QRhiShaderResourceBinding::VertexStage,
			                                              _uniform_buffer.get()),
			     QRhiShaderResourceBinding::sampledTexture(1,
			                                               QRhiShaderResourceBinding::VertexStage,
			                                               _rows_texture.get(), _sampler.get()),
			     QRhiShaderResourceBinding::sampledTexture(2,
			                                               QRhiShaderResourceBinding::VertexStage,
			                                               _colors_texture.get(),
			                                               _sampler.get())});
			if (not _bindings->create()) {
				_bindings.reset();
				return false;
			}
			_pipeline.reset();
		}

		if (_pipeline) {
			return true;
		}
		_pipeline.reset(_rhi->newGraphicsPipeline());
		// Everything is a vertical span, points included: one topology covers all three
		// draw modes.
		_pipeline->setTopology(QRhiGraphicsPipeline::Lines);
		_pipeline->setShaderStages({{QRhiShaderStage::Vertex, _vertex_shader},
		                            {QRhiShaderStage::Fragment, _fragment_shader}});
		// Nothing is fed in per vertex: the shader works out where it is from its index.
		_pipeline->setVertexInputLayout({});
		_pipeline->setShaderResourceBindings(_bindings.get());
		_pipeline->setRenderPassDescriptor(_render_pass.get());
		if (not _pipeline->create()) {
			_pipeline.reset();
			return false;
		}
		return true;
	}

#if QT_CONFIG(vulkan)
	std::unique_ptr<QVulkanInstance> _vulkan_instance;
#endif
	std::unique_ptr<QRhi> _rhi;
	std::unique_ptr<QRhiTexture> _colour_texture;
	std::unique_ptr<QRhiRenderPassDescriptor> _render_pass;
	std::unique_ptr<QRhiTextureRenderTarget> _render_target;
	std::unique_ptr<QRhiBuffer> _uniform_buffer;
	std::unique_ptr<QRhiTexture> _rows_texture;
	std::unique_ptr<QRhiTexture> _colors_texture;
	std::unique_ptr<QRhiSampler> _sampler;
	std::unique_ptr<QRhiShaderResourceBindings> _bindings;
	std::unique_ptr<QRhiGraphicsPipeline> _pipeline;

	QShader _vertex_shader;
	QShader _fragment_shader;
	std::vector<uchar> _rows_staging;
	std::vector<uchar> _colors_staging;
};

PVSeriesRendererQRhi::PVSeriesRendererQRhi(Squey::PVRangeSubSampler const& rss)
    : PVSeriesAbstractRenderer(rss)
{
}

PVSeriesRendererQRhi::~PVSeriesRendererQRhi() = default;

bool PVSeriesRendererQRhi::capability()
{
	// Bringing up a device is the only honest answer to "is there a GPU here", and it is
	// far too expensive to answer twice.
	static const bool available = [] {
		auto context = PVSeriesRhiContext::create();
		return context != nullptr;
	}();
	return available;
}

PVSeriesView::DrawMode PVSeriesRendererQRhi::capability(PVSeriesView::DrawMode mode)
{
	if (mode == PVSeriesView::DrawMode::Lines or mode == PVSeriesView::DrawMode::Points or
	    mode == PVSeriesView::DrawMode::LinesAlways) {
		return mode;
	}
	return PVSeriesView::DrawMode::Lines;
}

void PVSeriesRendererQRhi::set_background_color(QColor const& bgcol)
{
	_background_color = bgcol;
}

void PVSeriesRendererQRhi::set_draw_mode(PVSeriesView::DrawMode mode)
{
	_draw_mode = capability(mode);
}

QImage PVSeriesRendererQRhi::grab()
{
	const int w = width();
	const int h = height();
	if (w <= 0 or h <= 0) {
		return {};
	}

	const int samples_count =
	    _rss.valid() and not _series_draw_order.empty()
	        ? int(std::min<size_t>(
	              _rss.sampled_timeserie(_series_draw_order.front().dataIndex).size(), size_t(w)))
	        : 0;
	if (samples_count < 2) {
		QImage image(_size, QImage::Format_RGB32);
		image.fill(_background_color);
		return image;
	}

	if (not _context) {
		_context = PVSeriesRhiContext::create();
		if (not _context) {
			return {};
		}
	}

	PVSeriesSampleRows::project(_rss, _series_draw_order, _draw_mode, samples_count, h, _rows);

	QImage image = _context->render(_size, _rows, _series_draw_order, samples_count,
	                                _background_color, _draw_mode);
	if (image.isNull()) {
		// The device went away mid-session -- a driver reset, a laptop switching GPUs.
		// Dropping the context lets the next frame try again from scratch.
		_context.reset();
	}
	return image;
}

} // namespace PVParallelView
