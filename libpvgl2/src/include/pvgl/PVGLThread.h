//! \file PVCom.h
//! $Id: PVCom.h 3120 2011-06-14 05:16:22Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011


#ifndef PVGL_THREAD_H
#define PVGL_THREAD_H

#include <queue>

#include <QtCore>

#include <QThread>

#include <pvkernel/core/general.h>

#include <picviz/PVView.h>
#include <pvsdk/PVMessenger.h>

namespace PVGL {
/**
 * \class PVGLThread
 *
 * The main thread of the PVGL.
 */
class LibGLDecl PVGLThread : public QThread
{
	PVSDK::PVMessenger *pvsdk_messenger; //!<
public:
	/**
	 *  Constructor.
	 */
	PVGLThread () {pvsdk_messenger = new PVSDK::PVMessenger;}

	/**
	 * @return
	 */
	PVSDK::PVMessenger *get_messenger(){return pvsdk_messenger;}

	/**
	 *
	 */
	void run();

};
}
#endif	/* PVGL_THREAD_H */
