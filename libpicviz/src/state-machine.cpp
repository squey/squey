//! \file state-machine.cpp
//! $Id: state-machine.cpp 2865 2011-05-18 07:48:25Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/general.h>
#include <picviz/state-machine.h>

/******************************************************************************
 *
 * Picviz::StateMachine::StateMachine
 *
 *****************************************************************************/
Picviz::StateMachine::StateMachine()
{
	antialiased    = false;
	axes_mode      = false;
	sync_all_views = true;
	grabbed        = false;
	listing_unselected_visible = true;
	listing_zombie_visible     = true;
	unselected_visible         = true;
	zombie_visible             = true;
	edit_mode_all = true;
	square_area_mode = AREA_MODE_OFF;
}

/******************************************************************************
 *
 * Picviz::StateMachine::~StateMachine
 *
 *****************************************************************************/
Picviz::StateMachine::~StateMachine()
{
}
