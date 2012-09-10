#include <pvparallelview/PVBCIDrawingBackendCUDA.h>
#include <boost/thread.hpp>

PVParallelView::PVBCIDrawingEngineCuda* PVParallelView::PVBCIDrawingEngineCuda::_s = NULL;
boost::thread* PVParallelView::PVBCIDrawingEngineCuda::_thread;

PVParallelView::PVBCIDrawingEngineCuda::PVBCIDrawingEngineCuda():
		PVParallelView::PVBCIDrawingBackendSerial<PVParallelView::__impl::cuda_engine>()
{ }

PVParallelView::PVBCIDrawingEngineCuda* PVParallelView::PVBCIDrawingEngineCuda::get()
{
	if (!_s) {
		_s = new PVBCIDrawingEngineCuda();
		// Launch thread
		_thread = new boost::thread([=] { PVBCIDrawingEngineCuda::get()->run(); });
	}
	return _s;
}

void PVParallelView::PVBCIDrawingEngineCuda::release()
{
	if (_s) {
		// TODO: cancel waiting and all jobs
		delete _thread;
		delete _s;
	}
}
