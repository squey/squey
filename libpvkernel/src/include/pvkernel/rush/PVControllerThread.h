/**
 * \file PVControllerThread.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVRUSH_PVCONTROLLERTHREAD_FILE_H
#define PVRUSH_PVCONTROLLERTHREAD_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVController.h>

#include <boost/thread.hpp>

#include <QThread>

namespace PVRush {

class LibKernelDecl PVControllerThread: public QThread
{
public:
	PVControllerThread(PVController& ctrl);
	//PVControllerThread(PVControllerThread const& o);
	//~PVControllerThread();

public:
	//void start() { exec(); }
	//void wait();

protected:
	void run() { _ctrl(); }

protected:
	PVController& _ctrl;
	//boost::thread *_thread;
};

}

#endif
