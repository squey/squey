//! \file PVStateMachine.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVSTATEMACHINE_H
#define PICVIZ_PVSTATEMACHINE_H

#include <picviz/general.h>
#include <picviz/square-area.h>

namespace Picviz {

        class LibExport PVStateMachine {
            bool antialiased; // FIXME remove me! [STR: Why?]
            bool axes_mode;
            bool sync_all_views;
            bool grabbed;
            bool edit_mode_all;

	    /* lines states: this must *NOT* be handled by an enumeration
	     * as it will limits the different possibilities (enum max elements) 
	     * we have with lines in the future and makes more complex the way 
	     * we know about the given state we are.
	     */
            bool listing_unselected_visible;
            bool listing_zombie_visible;
            bool gl_unselected_visible;
            bool gl_zombie_visible;
        public:

	    /** Square Area Modes
	     * helps to identify the selection mode
	     * that is done with the square area
	     */
            enum SquareAreaModes {
                AREA_MODE_OFF,	                /**< No selection area */
                AREA_MODE_SET_WITH_VOLATILE,
                AREA_MODE_ADD_VOLATILE,
                AREA_MODE_SUBSTRACT_VOLATILE,
                AREA_MODE_INTERSECT_VOLATILE
            };

        private:
            SquareAreaModes square_area_mode;
            picviz_square_area_t *square_area;

        public:
            PVStateMachine();

            bool is_antialiased()const {
                return antialiased;
            }

            bool is_axes_mode()const {
                return axes_mode;
            }

            bool is_synced()const {
                return sync_all_views;
            }

            bool is_grabbed()const {
                return grabbed;
            }

            bool are_listing_all_visible()const {
		    return listing_unselected_visible && listing_zombie_visible;
	    }

            bool are_listing_none_visible()const {
		    return !(listing_unselected_visible || listing_zombie_visible);
	    }

            bool are_listing_unselected_visible()const {
                return listing_unselected_visible;
            }

            bool are_listing_zombie_visible()const {
                return listing_zombie_visible;
            }

            bool are_gl_all_visible()const {
		    return gl_unselected_visible && gl_zombie_visible;
	    }

            bool are_gl_none_visible()const {
		    return !(gl_unselected_visible || gl_zombie_visible);
	    }

            bool are_gl_unselected_visible()const {
                return gl_unselected_visible;
            }

            bool are_gl_zombie_visible()const {
                return gl_zombie_visible;
            }
            
            bool is_edit_mode_all()const {
                return edit_mode_all;
            }

            picviz_square_area_t *get_square_area()const {
                return square_area;
            }

            SquareAreaModes get_square_area_mode()const {
                return square_area_mode;
            }

            void set_antialiased(bool state) {
                antialiased = state;
            }

            void set_axes_mode(bool state) {
                axes_mode = state;
            }

            void set_synced(bool state) {
                sync_all_views = state;
            }

            void set_grabbed(bool state) {
                grabbed = state;
            }

            void set_listing_unselected_visible(bool visible) {
		    listing_unselected_visible = visible;
            }

            void set_listing_zombie_visible(bool visible) {
		    listing_zombie_visible = visible;
            }

            void set_gl_unselected_visible(bool visible) {
		    gl_unselected_visible = visible;
            }

            void set_gl_zombie_visibile(bool visible) {
		    gl_zombie_visible = visible;
            }

            void set_edit_mode_all(bool state) {
                edit_mode_all = state;
            }

            void set_square_area_mode(SquareAreaModes mode) {
                square_area_mode = mode;
            }

            void toggle_antialiased() {
                antialiased = !antialiased;
            }

            void toggle_axes_mode() {
                axes_mode = !axes_mode;
            }

            void toggle_synced() {
                sync_all_views = !sync_all_views;
            }

            void toggle_grabbed() {
                grabbed = !grabbed;
            }

            void toggle_listing_unselected_visibility() {
                listing_unselected_visible = !listing_unselected_visible;
            }

            void toggle_listing_zombie_visibility() {
                listing_zombie_visible = !listing_zombie_visible;
            }

            void toggle_gl_unselected_visibility() {
                gl_unselected_visible = !gl_unselected_visible;
            }

            void toggle_gl_zombie_visibility() {
                gl_zombie_visible = !gl_zombie_visible;
            }

            void toggle_edit_mode() {
                edit_mode_all = !edit_mode_all;
            }
            
            //void update_listing_mode();
        };

}

#endif /* PICVIZ_PVSTATEMACHINE_H */
