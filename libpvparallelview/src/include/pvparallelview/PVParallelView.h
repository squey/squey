#ifndef PVPARALLELVIEW_PVPARALLELVIEW_H
#define PVPARALLELVIEW_PVPARALLELVIEW_H

#include <picviz/PVPlotted.h>
#include <picviz/PVView_types.h>

#include <pvparallelview/PVBCIDrawingBackend.h>

#include <tbb/mutex.h>

namespace PVParallelView {

typedef PVBCIDrawingBackend<PARALLELVIEW_ZT_BBITS>  backend_full_t;
typedef PVBCIDrawingBackend<PARALLELVIEW_ZZT_BBITS> backend_zoom_t;

class PVLibView;

namespace __impl {

class PVParallelView: boost::noncopyable
{
	typedef std::map<Picviz::PVView*, PVLibView*> map_lib_views;

private:
	PVParallelView();

public:
	~PVParallelView();

public:
	static PVParallelView* get();
	static void release();

public:
	template <template <size_t> class Backend>
	void init_backends()
	{
		_backend_full = static_cast<backend_full_t*>(new Backend<PARALLELVIEW_ZT_BBITS>());
		_backend_zoom = static_cast<backend_zoom_t*>(new Backend<PARALLELVIEW_ZZT_BBITS>());
	}
	PVLibView* get_lib_view(Picviz::PVView& view);
	PVLibView* get_lib_view(Picviz::PVView& view, Picviz::PVPlotted::uint_plotted_table_t const& plotted, PVRow nrows, PVCol ncols);

	void remove_lib_view(Picviz::PVView& view);

	backend_full_t& backend_full() const { assert(_backend_full); return *_backend_full; }
	backend_zoom_t& backend_zoom() const { assert(_backend_zoom); return *_backend_zoom; }

	QColor const& color_view_bg() const { return _color_view_bg; }

private:
	backend_full_t* _backend_full;
	backend_zoom_t* _backend_zoom;

	map_lib_views _lib_views;
	tbb::mutex _mutex;

	QColor _color_view_bg;

private:
	static PVParallelView* _s;
};

}

namespace common {

	// Proxy functions
	template <template <size_t> class Backend>
		inline void init() { PVParallelView::__impl::PVParallelView::get()->init_backends<Backend>(); }

	void init_cuda();

	inline void remove_lib_view(Picviz::PVView& view) { PVParallelView::__impl::PVParallelView::get()->remove_lib_view(view); }
	inline PVLibView* get_lib_view(Picviz::PVView& view) { return PVParallelView::__impl::PVParallelView::get()->get_lib_view(view); }
	inline PVLibView* get_lib_view(Picviz::PVView& view, Picviz::PVPlotted::uint_plotted_table_t const& plotted, PVRow nrows, PVCol ncols) { return PVParallelView::__impl::PVParallelView::get()->get_lib_view(view, plotted, nrows, ncols); }
	inline void release() { PVParallelView::__impl::PVParallelView::release(); }
	inline backend_zoom_t& backend_zoom() { return PVParallelView::__impl::PVParallelView::get()->backend_zoom(); }
	inline backend_full_t& backend_full() { return PVParallelView::__impl::PVParallelView::get()->backend_full(); }
	inline QColor const& color_view_bg() { return PVParallelView::__impl::PVParallelView::get()->color_view_bg(); }
}

}

#endif