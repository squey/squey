#ifndef PVRUSH_PVCONTROLLERTHREAD_FILE_H
#define PVRUSH_PVCONTROLLERTHREAD_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVController.h>

#include <QThread>

namespace PVRush {

class LibKernelDecl PVControllerThread : public QThread {
public:
	PVControllerThread(PVController& ctrl);
public:
	void run();
protected:
	PVController& _ctrl;
};

}

#endif
