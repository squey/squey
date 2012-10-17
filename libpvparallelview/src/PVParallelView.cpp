#include <picviz/PVView.h>
#include <pvparallelview/PVBCIDrawingBackend.h>
#include <pvparallelview/PVBCIDrawingBackendCUDA.h>
#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>

PVParallelView::__impl::PVParallelView* PVParallelView::__impl::PVParallelView::_s = nullptr;

PVParallelView::__impl::PVParallelView::PVParallelView():
	_backend_full(nullptr),
	_backend_zoom(nullptr)
{
	const float win_r = pvconfig.value("pvgl/window_r", 0.2f).toFloat();
	const float win_g = pvconfig.value("pvgl/window_g", 0.2f).toFloat();
	const float win_b = pvconfig.value("pvgl/window_b", 0.2f).toFloat();
	const float win_a = pvconfig.value("pvgl/window_a", 1.0f).toFloat();

	_color_view_bg.setRgbF(win_r, win_g, win_b, win_a);
}

PVParallelView::__impl::PVParallelView::~PVParallelView()
{
	if (_backend_full) {
		delete _backend_full;
	}

	if (_backend_zoom) {
		delete _backend_zoom;
	}

	tbb::mutex::scoped_lock lock(_mutex);

	map_lib_views::iterator it;
	for (it = _lib_views.begin(); it != _lib_views.end(); it++) {
		delete it->second;
	}
}

PVParallelView::__impl::PVParallelView* PVParallelView::__impl::PVParallelView::get()
{
	if (_s == NULL) {
		_s = new PVParallelView();
	}
	return _s;
}

void PVParallelView::__impl::PVParallelView::release()
{
	if (_s) {
		delete _s;
	}
}

PVParallelView::PVLibView* PVParallelView::__impl::PVParallelView::get_lib_view(Picviz::PVView& view)
{
	tbb::mutex::scoped_lock lock(_mutex);

	map_lib_views::iterator it = _lib_views.find(&view);
	if (it != _lib_views.end()) {
		return it->second;
	}

	Picviz::PVView_sp view_sp = view.shared_from_this();
	PVLibView* new_view = new PVLibView(view_sp);
	_lib_views.insert(std::make_pair(&view, new_view));
	return new_view;
}

PVParallelView::PVLibView* PVParallelView::__impl::PVParallelView::get_lib_view(Picviz::PVView& view, Picviz::PVPlotted::uint_plotted_table_t const& plotted, PVRow nrows, PVCol ncols)
{
	tbb::mutex::scoped_lock lock(_mutex);

	map_lib_views::iterator it = _lib_views.find(&view);
	if (it != _lib_views.end()) {
		return it->second;
	}

	Picviz::PVView_sp view_sp = view.shared_from_this();
	PVLibView* new_view = new PVLibView(view_sp, plotted, nrows, ncols);
	_lib_views.insert(std::make_pair(&view, new_view));
	return new_view;
}

void PVParallelView::__impl::PVParallelView::remove_lib_view(Picviz::PVView& view)
{
	tbb::mutex::scoped_lock lock(_mutex);

	map_lib_views::iterator it = _lib_views.find(&view);
	if (it != _lib_views.end()) {
		delete it->second;
		_lib_views.erase(it);
	}
}

void PVParallelView::common::init_cuda()
{
	init<PVBCIDrawingBackendCUDA>();
}
