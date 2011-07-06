#ifndef PVRUSH_PVCONTROLLERTHREAD_FILE_H
#define PVRUSH_PVCONTROLLERTHREAD_FILE_H

#include <pvcore/general.h>
#include <pvrush/PVController.h>

#include <QThread>

namespace PVRush {

class LibExport PVControllerThread : public QThread {
public:
	PVControllerThread(PVController& ctrl);
public:
	void run();
protected:
	PVController& _ctrl;
};

}

#endif
