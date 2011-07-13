//! \file state-machine.cpp
//! $Id: state-machine.cpp 3251 2011-07-06 11:51:57Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/general.h>

#include <picviz/PVSelectionDisplay.h>
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

    unselected_visible = true;
    zombie_visible = true;

    listing_selection_mode = PVSelectionDisplay::ALL;
    gl_selection_mode = PVSelectionDisplay::ALL;
}


/******************************************************************************
 *
 * Picviz::PVStateMachine::~PVStateMachine
 *
 *****************************************************************************/
Picviz::PVStateMachine::~PVStateMachine() 
{
}

/******************************************************************************
 *
 * Picviz::PVStateMachine::are_listing_unselected_visible()
 *
 *****************************************************************************/
bool Picviz::PVStateMachine::are_listing_unselected_visible() 
{
	if ((listing_selection_mode & PVSelectionDisplay::NO_UNSELECTED) == PVSelectionDisplay::NO_UNSELECTED) {
		return false;
	}

	return true;
}

/******************************************************************************
 *
 * Picviz::PVStateMachine::are_listing_zombie_visible()
 *
 *****************************************************************************/
bool Picviz::PVStateMachine::are_listing_zombie_visible() 
{	
	if ((listing_selection_mode & PVSelectionDisplay::NO_ZOMBIES) == PVSelectionDisplay::NO_ZOMBIES) {
		return false;
	}

	return true;
}
/******************************************************************************
 *
 * Picviz::PVStateMachine::set_listing_zombie_visible
 *
 *****************************************************************************/
void Picviz::PVStateMachine::set_listing_zombie_visible(bool visible) 
{
	if (listing_selection_mode == PVSelectionDisplay::ALL) {
		if (!visible) {
			listing_selection_mode = PVSelectionDisplay::NO_ZOMBIES;
		}
	}

	if (listing_selection_mode == PVSelectionDisplay::NO_ZOMBIES) {
		if (visible) {
			listing_selection_mode = PVSelectionDisplay::ALL;
		}
	}

	if (listing_selection_mode == PVSelectionDisplay::NO_UNSELECTED) {
		if (!visible) {
			listing_selection_mode = static_cast<PVSelectionDisplay::PVSelectionDisplayMode_t>(PVSelectionDisplay::NO_UNSELECTED | PVSelectionDisplay::NO_ZOMBIES);
		}
	}
}


/******************************************************************************
 *
 * Picviz::PVStateMachine::
 *
 *****************************************************************************/
void Picviz::PVStateMachine::set_listing_unselected_visible(bool visible) 
{
	if (listing_selection_mode == PVSelectionDisplay::ALL) {
		if (!visible) {
			listing_selection_mode = PVSelectionDisplay::NO_UNSELECTED;
		}
	}

	if (listing_selection_mode == PVSelectionDisplay::NO_ZOMBIES) {
		if (!visible) {
			listing_selection_mode = static_cast<PVSelectionDisplay::PVSelectionDisplayMode_t>(PVSelectionDisplay::NO_UNSELECTED | PVSelectionDisplay::NO_ZOMBIES);
		}
	}

	if (listing_selection_mode == (PVSelectionDisplay::NO_UNSELECTED) | (PVSelectionDisplay::NO_ZOMBIES)) {
		if (visible) {
			listing_selection_mode = PVSelectionDisplay::NO_ZOMBIES;
		}
	}

	if (listing_selection_mode == PVSelectionDisplay::NO_UNSELECTED) {
		if (visible) {
			listing_selection_mode = PVSelectionDisplay::ALL;
		}
	}

}

/******************************************************************************
 *
 * Picviz::PVStateMachine::
 *
 *****************************************************************************/
void Picviz::PVStateMachine::toggle_listing_unselected_visibility() 
{
	switch (listing_selection_mode) {
	case PVSelectionDisplay::ALL:
                listing_selection_mode = PVSelectionDisplay::NO_UNSELECTED;
		break;
	case PVSelectionDisplay::NO_ZOMBIES:
                listing_selection_mode = static_cast<PVSelectionDisplay::PVSelectionDisplayMode_t>(PVSelectionDisplay::NO_UNSELECTED | PVSelectionDisplay::NO_ZOMBIES);
		break;
	case PVSelectionDisplay::NO_UNSELECTED:
                listing_selection_mode = PVSelectionDisplay::ALL;
		break;
	case PVSelectionDisplay::NO_ZOMBIES | PVSelectionDisplay::NO_UNSELECTED:
                listing_selection_mode = PVSelectionDisplay::NO_ZOMBIES;
		break;
	default:
		PVLOG_ERROR("Unknown listing mode!\n");
		break;
	}
}

/******************************************************************************
 *
 * Picviz::PVStateMachine::
 *
 *****************************************************************************/
void Picviz::PVStateMachine::toggle_listing_zombie_visibility() 
{
	switch (listing_selection_mode) {
	case PVSelectionDisplay::ALL:
                listing_selection_mode = PVSelectionDisplay::NO_ZOMBIES;
		break;
	case PVSelectionDisplay::NO_ZOMBIES:
                listing_selection_mode = PVSelectionDisplay::ALL;
		break;
	case PVSelectionDisplay::NO_UNSELECTED:
                listing_selection_mode = static_cast<PVSelectionDisplay::PVSelectionDisplayMode_t>(PVSelectionDisplay::NO_UNSELECTED | PVSelectionDisplay::NO_ZOMBIES);
		break;
	case PVSelectionDisplay::NO_ZOMBIES | PVSelectionDisplay::NO_UNSELECTED:
                listing_selection_mode = PVSelectionDisplay::NO_UNSELECTED;
		break;
	default:
		PVLOG_ERROR("Unknown listing mode!\n");
		break;
	}
}
