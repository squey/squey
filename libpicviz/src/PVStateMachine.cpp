//! \file state-machine.cpp
//! $Id: state-machine.cpp 3251 2011-07-06 11:51:57Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/general.h>
#include <picviz/PVStateMachine.h>

#define print_listing_mode(mode) {switch(mode){\
case LISTING_ALL:PVLOG_INFO("StateMachine listing mode is now :LISTING_ALL\n");break;\
case LISTING_NO_ZOMBIES:PVLOG_INFO("StateMachine listing mode is now :LISTING_NO_ZOMBIES\n");break;\
case LISTING_NO_UNSEL_NO_ZOMBIES:PVLOG_INFO("StateMachine listing mode is now :LISTING_NO_UNSEL_NO_ZOMBIES\n");break;\
case LISTING_NO_UNSEL:PVLOG_INFO("StateMachine listing mode is now :LISTING_NO_UNSEL\n");break;\
}}


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
    //listing_unselected_visible = true;
    //listing_zombie_visible = true;
    unselected_visible = true;
    zombie_visible = true;
    edit_mode_all = true;
    square_area_mode = AREA_MODE_OFF;
    
    listing_mode = LISTING_ALL;
    // listing_mode = LISTING_NO_UNSEL_NO_ZOMBIES;

    //update_listing_mode();
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
 * Picviz::PVStateMachine::
 *
 *****************************************************************************/
bool Picviz::PVStateMachine::are_listing_unselected_visible() 
{
	if ((listing_mode==LISTING_ALL) || (listing_mode==LISTING_NO_ZOMBIES)){
		PVLOG_DEBUG("StateMachine are_listing_unselected_visible yes\n");
		return true;
	}

	PVLOG_DEBUG("StateMachine are_listing_unselected_visible no\n");        
	return false;
}

/******************************************************************************
 *
 * Picviz::PVStateMachine::
 *
 *****************************************************************************/
bool Picviz::PVStateMachine::are_listing_zombie_visible() 
{
    if(listing_mode==LISTING_ALL||listing_mode==LISTING_NO_UNSEL){
        PVLOG_DEBUG("StateMachine are_listing_zombie_visible yes\n"); 
        return true;
    }
    PVLOG_DEBUG("StateMachine are_listing_zombie_visible no\n"); 
    return false;
}
/******************************************************************************
 *
 * Picviz::PVStateMachine::
 *
 *****************************************************************************/
void Picviz::PVStateMachine::set_listing_zombie_visibility(bool state) 
{
    //listing_zombie_visible = state;
    //update_listing_mode();
    PVLOG_DEBUG("StateMachine set_listing_zombie_visibility\n");

    switch(listing_mode){
        case LISTING_ALL:
		if (!state) {
			listing_mode = LISTING_NO_ZOMBIES;
		}
            break;
        case LISTING_NO_ZOMBIES:
		if (state) {
			listing_mode = LISTING_ALL;
		}
            break;
        case LISTING_NO_UNSEL_NO_ZOMBIES:
		if (state) {
			listing_mode = LISTING_NO_UNSEL;
		}
            break;
        case LISTING_NO_UNSEL:
		if (!state) {
			listing_mode = LISTING_NO_UNSEL_NO_ZOMBIES;
		}
            break;
        default:
		PVLOG_ERROR("Unknown listing mode!\n");
		break;
    }

    print_listing_mode(listing_mode);

}


/******************************************************************************
 *
 * Picviz::PVStateMachine::
 *
 *****************************************************************************/
void Picviz::PVStateMachine::set_listing_unselected_visibility(bool state) 
{
    //listing_unselected_visible = state;
    //update_listing_mode();
    PVLOG_DEBUG("StateMachine set_listing_unselected_visibility\n");
    switch(listing_mode){
        case LISTING_ALL:
		if(!state) {
			listing_mode = LISTING_NO_UNSEL;
		}
		break;
        case LISTING_NO_ZOMBIES:
		if(!state) {
			listing_mode = LISTING_NO_UNSEL_NO_ZOMBIES;
		}
		break;
        case LISTING_NO_UNSEL_NO_ZOMBIES:
		if(state) {
			listing_mode = LISTING_NO_ZOMBIES;
		}
		break;
        case LISTING_NO_UNSEL:
		if(state) {
			listing_mode=LISTING_ALL;
		}
		break;
        default:
		PVLOG_ERROR("Unknown listing mode!\n");
		break;
    }
    print_listing_mode(listing_mode);
}

/******************************************************************************
 *
 * Picviz::PVStateMachine::
 *
 *****************************************************************************/
void Picviz::PVStateMachine::toggle_listing_unselected_visibility() 
{
    //listing_unselected_visible = !listing_unselected_visible;
    //update_listing_mode();
    PVLOG_DEBUG("StateMachine toggle_listing_unselected_visibility\n");
    switch(listing_mode){
        case LISTING_ALL:
                listing_mode = LISTING_NO_UNSEL;
            break;
        case LISTING_NO_ZOMBIES:
                listing_mode = LISTING_NO_UNSEL_NO_ZOMBIES;
            break;
        case LISTING_NO_UNSEL_NO_ZOMBIES:
                listing_mode = LISTING_NO_ZOMBIES;
            break;
        case LISTING_NO_UNSEL:
                listing_mode = LISTING_ALL;
            break;
        default:
		PVLOG_ERROR("Unknown listing mode!\n");
		break;
    }
    print_listing_mode(listing_mode);
}

/******************************************************************************
 *
 * Picviz::PVStateMachine::
 *
 *****************************************************************************/
void Picviz::PVStateMachine::toggle_listing_zombie_visibility() 
{
    //listing_zombie_visible = !listing_zombie_visible;
    //update_listing_mode();
    PVLOG_DEBUG("StateMachine toggle_listing_zombie_visibility\n");
    switch (listing_mode) {
        case LISTING_ALL:
            listing_mode = LISTING_NO_ZOMBIES;
            break;
        case LISTING_NO_ZOMBIES:
            listing_mode = LISTING_ALL;
            break;
        case LISTING_NO_UNSEL_NO_ZOMBIES:
            listing_mode = LISTING_NO_UNSEL;
            break;
        case LISTING_NO_UNSEL:
            listing_mode = LISTING_NO_UNSEL_NO_ZOMBIES;
            break;
        default:
		PVLOG_ERROR("Unknown listing mode!\n");
		break;
    }
    print_listing_mode(listing_mode);
}


