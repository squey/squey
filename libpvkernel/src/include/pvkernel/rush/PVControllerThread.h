#ifndef PVRUSH_PVCONTROLLERTHREAD_FILE_H
#define PVRUSH_PVCONTROLLERTHREAD_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVController.h>

#include <tbb/compat/thread>

namespace PVRush {

class LibKernelDecl PVControllerThread
{
public:
	PVControllerThread(PVController& ctrl);
public:
	void start();
	void wait();
protected:
	PVController& _ctrl;
	std::thread _thread;
};

}

#endif
