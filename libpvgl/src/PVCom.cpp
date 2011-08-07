//! \file PVCom.cpp
//! $Id: PVCom.cpp 2998 2011-05-29 11:15:58Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvgl/PVUtils.h>
#include <pvgl/PVCom.h>
#include <pvgl/general.h>

/******************************************************************************
 *
 * PVGL::PVCom::get_message_for_gl
 *
 *****************************************************************************/
bool PVGL::PVCom::get_message_for_gl(struct PVMessage &message)
{
	bool return_value;

	PVLOG_HEAVYDEBUG("PVGL::PVCom::%s\n", __FUNCTION__);

	mutex.lock();
	if (qt_to_gl.empty()) {
		return_value = false;
	} else {
		message = qt_to_gl.front();
		qt_to_gl.pop();
		return_value = true;
	}
	mutex.unlock();
	return return_value;
}

/******************************************************************************
 *
 * PVGL::PVCom::get_message_for_qt
 *
 *****************************************************************************/
bool PVGL::PVCom::get_message_for_qt(struct PVMessage &message)
{
	bool return_value;

	PVLOG_HEAVYDEBUG("PVGL::PVCom::%s\n", __FUNCTION__);

	mutex.lock();
	if (gl_to_qt.empty()) {
		return_value = false;
	} else {
		message = gl_to_qt.front();
		gl_to_qt.pop();
		return_value = true;
	}
	mutex.unlock();
	return return_value;
}

/******************************************************************************
 *
 * PVGL::PVCom::post_message_to_gl
 *
 *****************************************************************************/
void PVGL::PVCom::post_message_to_gl(const struct PVMessage &message)
{
	PVLOG_HEAVYDEBUG("PVGL::PVCom::%s\n", __FUNCTION__);

	mutex.lock();
	qt_to_gl.push(message);
	mutex.unlock();
}

/******************************************************************************
 *
 * PVGL::PVCom::post_message_to_qt
 *
 *****************************************************************************/
void PVGL::PVCom::post_message_to_qt(const struct PVMessage &message)
{
	PVLOG_HEAVYDEBUG("PVGL::PVCom::%s\n", __FUNCTION__);

	mutex.lock();
	gl_to_qt.push(message);
	mutex.unlock();
}

/******************************************************************************
 *
 * PVGL::PVThread::run
 *
 *****************************************************************************/
void PVGL::PVThread::run ()
{
	PVLOG_DEBUG("PVGL::PVThread::%s\n", __FUNCTION__);

	pvgl_init(pvgl_com);
}
