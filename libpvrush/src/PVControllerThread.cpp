#include <pvrush/PVControllerThread.h>


PVRush::PVControllerThread::PVControllerThread(PVController &ctrl) :
	_ctrl(ctrl)
{
}

void PVRush::PVControllerThread::run()
{
	PVLOG_DEBUG("Controller thread launched !\n");
	_ctrl();
	PVLOG_DEBUG("Controller thread stopped !\n");
}
