//! \file state-machine.cpp
//! $Id: state-machine.cpp 3251 2011-07-06 11:51:57Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/general.h>
#include <picviz/PVStateMachine.h>

/******************************************************************************
 *
 * Picviz::PVStateMachine::PVStateMachine
 *
 *****************************************************************************/
Picviz::PVStateMachine::PVStateMachine() 
{
	antialiased = false;
	axes_mode = false;
	sync_all_views = true;
	grabbed = false;
	edit_mode_all = true;
	square_area_mode = AREA_MODE_OFF;	

	listing_unselected_visible = false;
	listing_zombie_visible = false;
	gl_unselected_visible = true;
	gl_zombie_visible = true;

	caps_lock_activated = false;
}

QString Picviz::PVStateMachine::get_string()
{
	QString str("");

	if (listing_unselected_visible) {
		str.append("U");
	} else {
		str.append("NU");
	}
	if (listing_zombie_visible) {
		str.append("Z");
	} else {
		str.append("NZ");
	}
	str.append("|");
	if (gl_unselected_visible) {
		str.append("U");
	} else {
		str.append("NU");
	}
	if (gl_zombie_visible) {
		str.append("Z");
	} else {
		str.append("NZ");
	}

	return str;
}
