#include <pvkernel/rush/PVControllerThread.h>


PVRush::PVControllerThread::PVControllerThread(PVController &ctrl) :
	_ctrl(ctrl)
{
}

#if 0
PVRush::PVControllerThread::PVControllerThread(PVControllerThread const& o):
	_ctrl(o._ctrl), _thread(NULL)
{ }

PVRush::PVControllerThread::~PVControllerThread()
{
	if (_thread) {
		delete _thread;
	}
}

void PVRush::PVControllerThread::start()
{
	if (_thread && _thread->joinable()) {
		PVLOG_WARN("Controller thread already running !\n");
		return;
	}
	_thread = new boost::thread(boost::bind(&PVController::operator(), &_ctrl));
}

void PVRush::PVControllerThread::wait()
{
	if (_thread && _thread->joinable()) {
		_thread->join();
		delete _thread;
	}
}
#endif
