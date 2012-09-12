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
}

PVParallelView::__impl::PVParallelView::~PVParallelView()
{
	if (_backend_full) {
		delete _backend_full;
	}

	if (_backend_zoom) {
		delete _backend_zoom;
	}

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
	map_lib_views::iterator it = _lib_views.find(&view);
	if (it != _lib_views.end()) {
		return it->second;
	}

	Picviz::PVView_sp view_sp = view.shared_from_this();
	PVLibView* new_view = new PVLibView(view_sp, plotted, nrows, ncols);
	_lib_views.insert(std::make_pair(&view, new_view));
	return new_view;
}

void PVParallelView::common::init_cuda()
{
	init<PVBCIDrawingBackendCUDA>();
}
