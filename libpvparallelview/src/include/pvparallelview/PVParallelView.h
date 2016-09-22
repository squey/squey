/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVPARALLELVIEW_H
#define PVPARALLELVIEW_PVPARALLELVIEW_H

#include <inendi/PVPlotted.h>

#include <pvparallelview/PVBCIDrawingBackend.h>

#include <tbb/mutex.h>

namespace PVParallelView
{

class PVLibView;
class PVRenderingPipeline;

class PVParallelViewImpl
{
	typedef std::map<Inendi::PVView*, PVLibView*> map_lib_views;

  private:
	PVParallelViewImpl();
	PVParallelViewImpl(const PVParallelViewImpl&) = delete;

  public:
	~PVParallelViewImpl();

  public:
	static PVParallelViewImpl& get();

  public:
	template <class Backend>
	void init_backends()
	{
		_backend = static_cast<PVBCIDrawingBackend*>(&Backend::get());
		init_pipeline();
		register_displays();
	}

	PVLibView* get_lib_view(Inendi::PVView& view);
	PVLibView* get_lib_view(Inendi::PVView& view,
	                        Inendi::PVPlotted::uint_plotted_table_t const& plotted,
	                        PVRow nrows,
	                        PVCol ncols);

	void remove_lib_view(Inendi::PVView& view);

	PVBCIDrawingBackend& backend() const
	{
		assert(_backend);
		return *_backend;
	}
	PVRenderingPipeline& pipeline() const
	{
		assert(_pipeline);
		return *_pipeline;
	}

	QColor const& color_view_bg() const { return _color_view_bg; }

#ifdef INENDI_DEVELOPER_MODE
	bool show_bboxes() const { return _show_bboxes; }
	void toggle_show_bboxes() { _show_bboxes = !_show_bboxes; }
#endif

  private:
	void init_pipeline();
	void register_displays();

  private:
	PVParallelView::PVBCIDrawingBackend* _backend;
	// For compile-time sake, PVRenderingPipeline is not included and will be heap-allocated.
	// This is just done once, so no real issue here.. !
	PVParallelView::PVRenderingPipeline* _pipeline;

	map_lib_views _lib_views;
	tbb::mutex _mutex;

	QColor _color_view_bg;

	// This is used in developer mode to tell whether bounding boxes should be visible
	// This is let even if non developer mode not to change the size of this structure...
	bool _show_bboxes;

  private:
	static PVParallelViewImpl* _s; //<! Instance of the singleton
};

namespace common
{

/**
 * RAII class to automatically free ParallelView backend resources.
 *
 * We need a specific RAII class here instead of scott meyer's singleton as
 * static variables are free'd after global variables so nvidia driver is shutdown
 * before memory disallocation and disallocation will fail.
 */
class RAII_backend_init
{
  public:
	RAII_backend_init();

	~RAII_backend_init() { delete _instance; }

  private:
	PVParallelView::PVParallelViewImpl* _instance; // Singleton pointer of the ParallelViewImpl.
};

// Proxy functions
inline void remove_lib_view(Inendi::PVView& view)
{
	PVParallelView::PVParallelViewImpl::get().remove_lib_view(view);
}
inline PVLibView* get_lib_view(Inendi::PVView& view)
{
	return PVParallelView::PVParallelViewImpl::get().get_lib_view(view);
}
inline PVLibView* get_lib_view(Inendi::PVView& view,
                               Inendi::PVPlotted::uint_plotted_table_t const& plotted,
                               PVRow nrows,
                               PVCol ncols)
{
	return PVParallelView::PVParallelViewImpl::get().get_lib_view(view, plotted, nrows, ncols);
}
inline PVBCIDrawingBackend& backend()
{
	return PVParallelView::PVParallelViewImpl::get().backend();
}
inline PVRenderingPipeline& pipeline()
{
	return PVParallelView::PVParallelViewImpl::get().pipeline();
}
inline QColor const& color_view_bg()
{
	return PVParallelView::PVParallelViewImpl::get().color_view_bg();
}

inline bool is_gpu_accelerated()
{
	return PVParallelView::PVParallelViewImpl::get().backend().is_gpu_accelerated();
}

#ifdef INENDI_DEVELOPER_MODE
inline bool show_bboxes()
{
	return PVParallelView::PVParallelViewImpl::get().show_bboxes();
}
inline void toggle_show_bboxes()
{
	return PVParallelView::PVParallelViewImpl::get().toggle_show_bboxes();
}

#endif
} // namespace common
} // namespace PVParallelView

#endif
