/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSH_PVCONTROLLERTHREAD_FILE_H
#define PVRUSH_PVCONTROLLERTHREAD_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVController.h>

#include <boost/thread.hpp>

#include <QThread>

namespace PVRush {

class PVControllerThread: public QThread
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
