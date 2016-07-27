/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVStateMachine.h>

/******************************************************************************
 *
 * Inendi::PVStateMachine::PVStateMachine
 *
 *****************************************************************************/
Inendi::PVStateMachine::PVStateMachine()
{
	square_area_mode = AREA_MODE_OFF;

	listing_unselected_visible = false;
	listing_zombie_visible = false;
}

QString Inendi::PVStateMachine::get_string()
{
	QString str;

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

	return str;
}
