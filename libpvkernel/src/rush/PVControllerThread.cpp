#include <pvkernel/rush/PVControllerThread.h>


PVRush::PVControllerThread::PVControllerThread(PVController &ctrl) :
	_ctrl(ctrl)
{
}

void PVRush::PVControllerThread::start()
{
	if (_thread.joinable()) {
		PVLOG_WARN("Controller thread already running !\n");
		return;
	}
	std::thread th(boost::bind(&PVController::operator(), &_ctrl));
	_thread = th;
}

void PVRush::PVControllerThread::wait()
{
	if (_thread.joinable()) {
		_thread.join();
	}
}
