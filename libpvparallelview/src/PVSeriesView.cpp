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
			_renderer = std::make_unique<Renderer>(_rss);                                          \
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
    : QWidget(parent), _rss(rss), _backend(make_renderer(backend)), _pixmap(size())
{
	_pixmap.fill(Qt::black);

	_rss._subsampled.connect([this] { refresh(); });
}

PVSeriesView::~PVSeriesView() = default;

void PVSeriesView::set_background_color(QColor const& bgcol)
{
	setPalette(QPalette(bgcol));
	_renderer->set_background_color(bgcol);
}

void PVSeriesView::show_series(std::vector<SerieDrawInfo> series_draw_order)
{
	_renderer->show_series(std::move(series_draw_order));
}

auto PVSeriesView::capability(Backend backend, DrawMode draw_mode) -> DrawMode
{
	SWITCH_BACKEND(backend, draw_mode)
}

void PVSeriesView::set_draw_mode(DrawMode dm)
{
	_renderer->set_draw_mode(dm);
}

void PVSeriesView::refresh()
{
	_need_hard_redraw = true;
	update();
}

void PVSeriesView::paintEvent(QPaintEvent*)
{
	if (_need_hard_redraw) {
		BENCH_START(paint_series);
		_pixmap = _renderer->grab();
		BENCH_END(paint_series, "paint_series", 1, 1, 1, 1);
		_need_hard_redraw = false;
	}
	QPainter painter(this);
	painter.drawPixmap(0, 0, width(), height(), _pixmap);
}

void PVSeriesView::resizeEvent(QResizeEvent*)
{
	_renderer->resize(size());
}

} // namespace PVParallelView
