#ifndef PVPARALLELVIEW_PVPARALLELVIEW_H
#define PVPARALLELVIEW_PVPARALLELVIEW_H

#include <picviz/PVPlotted.h>
#include <picviz/PVView_types.h>

#include <pvparallelview/PVBCIDrawingBackend.h>

#include <tbb/mutex.h>

namespace PVParallelView {

class PVLibView;
class PVRenderingPipeline;

class PVParallelViewImpl: boost::noncopyable
{
	typedef std::map<Picviz::PVView*, PVLibView*> map_lib_views;

private:
	PVParallelViewImpl();

public:
	~PVParallelViewImpl();

public:
	static PVParallelViewImpl* get();
	static void release();

public:
	template <class Backend>
	void init_backends()
	{
		_backend = static_cast<PVBCIDrawingBackend*>(&Backend::get());
		init_pipeline();
	}
	PVLibView* get_lib_view(Picviz::PVView& view);
	PVLibView* get_lib_view(Picviz::PVView& view, Picviz::PVPlotted::uint_plotted_table_t const& plotted, PVRow nrows, PVCol ncols);

	void remove_lib_view(Picviz::PVView& view);

	PVBCIDrawingBackend& backend() const { assert(_backend); return *_backend; }
	PVRenderingPipeline& pipeline() const { assert(_pipeline); return *_pipeline; }

	QColor const& color_view_bg() const { return _color_view_bg; }

private:
	void init_pipeline();

private:
	PVParallelView::PVBCIDrawingBackend* _backend;
	// For compile-time sake, PVRenderingPipeline is not included and will be heap-allocated.
	// This is just done once, so no real issue here.. !
	PVParallelView::PVRenderingPipeline* _pipeline;

	map_lib_views _lib_views;
	tbb::mutex _mutex;

	QColor _color_view_bg;

private:
	static PVParallelViewImpl* _s;
};

namespace common {

	// Proxy functions
	template <class Backend>
	inline void init() { PVParallelView::PVParallelViewImpl::get()->init_backends<Backend>(); }

	void init_cuda();

	inline void remove_lib_view(Picviz::PVView& view) { PVParallelView::PVParallelViewImpl::get()->remove_lib_view(view); }
	inline PVLibView* get_lib_view(Picviz::PVView& view) { return PVParallelView::PVParallelViewImpl::get()->get_lib_view(view); }
	inline PVLibView* get_lib_view(Picviz::PVView& view, Picviz::PVPlotted::uint_plotted_table_t const& plotted, PVRow nrows, PVCol ncols) { return PVParallelView::PVParallelViewImpl::get()->get_lib_view(view, plotted, nrows, ncols); }
	inline void release() { PVParallelView::PVParallelViewImpl::release(); }
	inline PVBCIDrawingBackend& backend() { return PVParallelView::PVParallelViewImpl::get()->backend(); }
	inline PVRenderingPipeline& pipeline() { return PVParallelView::PVParallelViewImpl::get()->pipeline(); }
	inline QColor const& color_view_bg() { return PVParallelView::PVParallelViewImpl::get()->color_view_bg(); }
}

}

#endif
