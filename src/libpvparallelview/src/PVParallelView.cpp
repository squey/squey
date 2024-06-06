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

#include <pvkernel/core/PVConfig.h>

#include <squey/PVView.h>
#include <pvparallelview/PVBCIDrawingBackend.h>
#include <pvparallelview/PVBCIDrawingBackendOpenCL.h>
#include <pvparallelview/PVBCIDrawingBackendQPainter.h>
#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVRenderingPipeline.h>
#include <pvparallelview/PVZoneRendering.h>

// Displays plugins
#include <pvparallelview/PVDisplayViewFullParallel.h>
#include <pvparallelview/PVDisplayViewZoomedParallel.h>
#include <pvparallelview/PVDisplayViewHitCount.h>
#include <pvparallelview/PVDisplayViewScatter.h>
#include <pvparallelview/PVDisplayViewTimeseries.h>

#include <pvkernel/core/PVTheme.h>

#include <QSettings>

PVParallelView::PVParallelViewImpl* PVParallelView::PVParallelViewImpl::_s = nullptr;

PVParallelView::PVParallelViewImpl::PVParallelViewImpl()
    : _backend(nullptr), _pipeline(nullptr), _show_bboxes(false)
{
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
	REGISTER_CLASS("parallelview_timeseriesview", PVDisplays::PVDisplayViewTimeseries);
}

PVParallelView::PVParallelViewImpl& PVParallelView::PVParallelViewImpl::get()
{
	if (not _s) {
		_s = new PVParallelViewImpl;
	}
	return *_s;
}

PVParallelView::PVLibView* PVParallelView::PVParallelViewImpl::get_lib_view(Squey::PVView& view)
{
	tbb::mutex::scoped_lock lock(_mutex);

	auto it = _lib_views.find(&view);
	if (it != _lib_views.end()) {
		return it->second;
	}

	auto* new_view = new PVLibView(view);
	_lib_views.insert(std::make_pair(&view, new_view));
	return new_view;
}

void PVParallelView::PVParallelViewImpl::remove_lib_view(Squey::PVView& view)
{
	tbb::mutex::scoped_lock lock(_mutex);

	auto it = _lib_views.find(&view);
	if (it != _lib_views.end()) {
		delete it->second;
		_lib_views.erase(it);
	}
}

namespace PVParallelView::common
{
/************************************************************
 *
 * RAII backend resources implementation
 *
 ************************************************************/
RAII_backend_init::RAII_backend_init() : _instance(&PVParallelView::PVParallelViewImpl::get())
{
	if (PVBCIDrawingBackendOpenCL::get().device_count() > 0) {
		_instance->init_backends<PVBCIDrawingBackendOpenCL>();
	} else {
		_instance->init_backends<PVBCIDrawingBackendQPainter>();
	}
}
} // namespace PVParallelView
