//! \file PVMessenger.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvsdk/PVMessenger.h>
#include <pvsdk/general.h>

/******************************************************************************
 *
 * PVSDK::PVMessenger::get_message_for_gl
 *
 *****************************************************************************/
bool PVSDK::PVMessenger::get_message_for_gl(struct PVMessage &message)
{
	bool return_value;

	PVLOG_HEAVYDEBUG("PVSDK::PVMessenger::%s\n", __FUNCTION__);

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
 * PVSDK::PVMessenger::get_message_for_qt
 *
 *****************************************************************************/
bool PVSDK::PVMessenger::get_message_for_qt(struct PVMessage &message)
{
	bool return_value;

	PVLOG_HEAVYDEBUG("PVSDK::PVMessenger::%s\n", __FUNCTION__);

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
 * PVSDK::PVMessenger::post_message_to_gl
 *
 *****************************************************************************/
void PVSDK::PVMessenger::post_message_to_gl(const struct PVMessage &message)
{
	PVLOG_HEAVYDEBUG("PVSDK::PVMessenger::%s\n", __FUNCTION__);

	mutex.lock();
	qt_to_gl.push(message);
	mutex.unlock();
}

/******************************************************************************
 *
 * PVSDK::PVMessenger::post_message_to_qt
 *
 *****************************************************************************/
void PVSDK::PVMessenger::post_message_to_qt(const struct PVMessage &message)
{
	PVLOG_HEAVYDEBUG("PVSDK::PVMessenger::%s\n", __FUNCTION__);

	mutex.lock();
	gl_to_qt.push(message);
	mutex.unlock();
}

