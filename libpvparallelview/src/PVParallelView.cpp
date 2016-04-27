/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVConfig.h>

#include <inendi/PVView.h>
#include <pvparallelview/PVBCIDrawingBackend.h>
#ifdef CUDA
#include <pvparallelview/PVBCIDrawingBackendCUDA.h>
#endif
#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVRenderingPipeline.h>
#include <pvparallelview/PVZoneRendering.h>

// Displays plugins
#include <pvparallelview/PVDisplayViewFullParallel.h>
#include <pvparallelview/PVDisplayViewZoomedParallel.h>
#include <pvparallelview/PVDisplayViewHitCount.h>
#include <pvparallelview/PVDisplayViewScatter.h>

PVParallelView::PVParallelViewImpl* PVParallelView::PVParallelViewImpl::_s = nullptr;

PVParallelView::PVParallelViewImpl::PVParallelViewImpl()
    : _backend(nullptr), _pipeline(nullptr), _show_bboxes(false)
{
	QSettings& pvconfig = PVCore::PVConfig::get().config();

	const float win_r = pvconfig.value("pvgl/window_r", 0.2f).toFloat();
	const float win_g = pvconfig.value("pvgl/window_g", 0.2f).toFloat();
	const float win_b = pvconfig.value("pvgl/window_b", 0.2f).toFloat();
	const float win_a = pvconfig.value("pvgl/window_a", 1.0f).toFloat();

	_color_view_bg.setRgbF(win_r, win_g, win_b, win_a);

	qRegisterMetaType<PVParallelView::PVZoneRendering_p>();
	qRegisterMetaType<PVParallelView::PVZoneRendering_p>("PVZoneRendering_p");
}

PVParallelView::PVParallelViewImpl::~PVParallelViewImpl()
{
	tbb::mutex::scoped_lock lock(_mutex);

	map_lib_views::iterator it;
	for (it = _lib_views.begin(); it != _lib_views.end(); it++) {
		delete it->second;
	}

	if (_pipeline) {
		delete _pipeline;
	}
}

void PVParallelView::PVParallelViewImpl::init_pipeline()
{
	if (_pipeline) {
		delete _pipeline;
	}
	_pipeline = new PVParallelView::PVRenderingPipeline(backend());
}

void PVParallelView::PVParallelViewImpl::register_displays()
{
	REGISTER_CLASS("parallelview_fullparallelview", PVDisplays::PVDisplayViewFullParallel);
	REGISTER_CLASS("parallelview_zoomedparallelview", PVDisplays::PVDisplayViewZoomedParallel);
	REGISTER_CLASS("parallelview_hitcountview", PVDisplays::PVDisplayViewHitCount);
	REGISTER_CLASS("parallelview_scatterview", PVDisplays::PVDisplayViewScatter);
}

PVParallelView::PVParallelViewImpl& PVParallelView::PVParallelViewImpl::get()
{
	if (not _s) {
		_s = new PVParallelViewImpl;
	}
	return *_s;
}

PVParallelView::PVLibView* PVParallelView::PVParallelViewImpl::get_lib_view(Inendi::PVView& view)
{
	tbb::mutex::scoped_lock lock(_mutex);

	map_lib_views::iterator it = _lib_views.find(&view);
	if (it != _lib_views.end()) {
		return it->second;
	}

	Inendi::PVView_sp view_sp = view.shared_from_this();
	PVLibView* new_view = new PVLibView(view_sp);
	_lib_views.insert(std::make_pair(&view, new_view));
	return new_view;
}

void PVParallelView::PVParallelViewImpl::remove_lib_view(Inendi::PVView& view)
{
	tbb::mutex::scoped_lock lock(_mutex);

	map_lib_views::iterator it = _lib_views.find(&view);
	if (it != _lib_views.end()) {
		delete it->second;
		_lib_views.erase(it);
	}
}

namespace PVParallelView
{
namespace common
{
/************************************************************
 *
 * RAII cuda resources implementation
 *
 ************************************************************/
RAII_cuda_init::RAII_cuda_init() : _instance(&PVParallelView::PVParallelViewImpl::get())
{
	_instance->init_backends<PVBCIDrawingBackendCUDA>();
}
}
}
