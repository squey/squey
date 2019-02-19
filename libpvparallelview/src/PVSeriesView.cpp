#include <pvparallelview/PVSeriesView.h>

#include <pvparallelview/PVSeriesRendererOffscreen.h>
#include <pvparallelview/PVSeriesRendererOpenGL.h>
#include <pvparallelview/PVSeriesRendererQPainter.h>

#include <cassert>
#include <stdexcept>
#include <QPainter>

namespace PVParallelView
{

#define SWITCH_BACKEND(backend, ...)                                                               \
	switch (backend) {                                                                             \
		CASELIST_BACKEND(__VA_ARGS__)                                                              \
	default:                                                                                       \
		throw std::invalid_argument("Unkown PVSeriesView::Backend");                               \
	}

#define CASELIST_BACKEND(...)                                                                      \
	CASE_BACKEND(Backend::OffscreenOpenGL, PVSeriesRendererOffscreen, __VA_ARGS__)                 \
	CASE_BACKEND(Backend::OpenGL, PVSeriesRendererOpenGL, __VA_ARGS__)                             \
	CASE_BACKEND(Backend::QPainter, PVSeriesRendererQPainter, __VA_ARGS__)

#define CASE_BACKEND(backend, Renderer, ...)                                                       \
	case backend:                                                                                  \
		if (Renderer::capability()) {                                                              \
			qDebug() << "Choosing " << #backend "( " #Renderer " )";                               \
			m_renderer = std::make_unique<Renderer>(m_rss);                                        \
			return backend;                                                                        \
		}                                                                                          \
		[[fallthrough]];

auto PVSeriesView::make_renderer(Backend backend) -> Backend
{
	SWITCH_BACKEND(backend)
}

#undef CASE_BACKEND
#define CASE_BACKEND(backend, Renderer, ...)                                                       \
	case backend:                                                                                  \
		if (Renderer::capability()) {                                                              \
			return backend;                                                                        \
		}                                                                                          \
		[[fallthrough]];

auto PVSeriesView::capability(Backend backend) -> Backend
{
	SWITCH_BACKEND(backend)
}

#undef CASE_BACKEND
#define CASE_BACKEND(backend, Renderer, test)                                                      \
	case backend:                                                                                  \
		return Renderer::capability(test);

PVSeriesView::PVSeriesView(Inendi::PVRangeSubSampler& rss, Backend backend, QWidget* parent)
    : QWidget(parent), m_rss(rss), m_backend(make_renderer(backend)), m_pixmap(size())
{
	m_pixmap.fill(Qt::black);

	m_rss._subsampled.connect([this] { refresh(); });
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
}

auto PVSeriesView::capability(Backend backend, DrawMode drawMode) -> DrawMode
{
	SWITCH_BACKEND(backend, drawMode)
}

void PVSeriesView::setDrawMode(DrawMode dm)
{
	m_renderer->setDrawMode(dm);
}

void PVSeriesView::refresh()
{
	m_needHardRedraw = true;
	update();
}

void PVSeriesView::paintEvent(QPaintEvent*)
{
	if (m_needHardRedraw) {
		// qDebug() << "hard paint";
		BENCH_START(paint_series);
		m_pixmap = m_renderer->grab();
		BENCH_END(paint_series, "paint_series", 1, 1, 1, 1);

		m_needHardRedraw = false;
	}
	// qDebug() << "soft paint";
	QPainter painter(this);
	painter.drawPixmap(0, 0, width(), height(), m_pixmap);
}

void PVSeriesView::resizeEvent(QResizeEvent*)
{
	m_renderer->resize(size());
}

} // namespace PVParallelView
