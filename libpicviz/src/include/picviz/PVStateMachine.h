//! \file PVStateMachine.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVSTATEMACHINE_H
#define PICVIZ_PVSTATEMACHINE_H

#include <picviz/general.h>
#include <picviz/square-area.h>

#include <picviz/PVSelectionDisplay.h>

namespace Picviz {

        class LibExport PVStateMachine {
            bool antialiased; // FIXME remove me!
            bool axes_mode;
            bool sync_all_views;
            bool grabbed;
            bool unselected_visible;
            bool zombie_visible;
            bool edit_mode_all;

	    PVSelectionDisplay::PVSelectionDisplayMode_t listing_selection_mode;
            PVSelectionDisplay::PVSelectionDisplayMode_t gl_selection_mode;
        public:

            enum SquareAreaModes {
                AREA_MODE_OFF,
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
            ~PVStateMachine();

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

            bool are_listing_unselected_visible();

            bool are_listing_zombie_visible();

	    PVSelectionDisplay::PVSelectionDisplayMode_t are_listing_mode()const {
                return listing_selection_mode;
            }

            bool are_unselected_visible()const {
                return unselected_visible;
            }

            bool are_zombie_visible()const {
                return zombie_visible;
            }
            
            PVStateMachineListingMode_t getListingMode(){
                return listing_mode;
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

            void set_listing_unselected_visible(bool visible);

            void set_listing_zombie_visible(bool visible);

            void set_listing_mode(PVSelectionDisplay::PVSelectionDisplayMode_t state) {
                listing_selection_mode = state;
            }

            void set_unselected_visibility(bool state) {
                unselected_visible = state;
            }

            void set_zombie_visibility(bool state) {
                zombie_visible = state;
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

            void toggle_listing_unselected_visibility();

            void toggle_listing_zombie_visibility();

            void toggle_unselected_visibility() {
                unselected_visible = !unselected_visible;
            }

            void toggle_zombie_visibility() {
                zombie_visible = !zombie_visible;
            }

            void toggle_edit_mode() {
                edit_mode_all = !edit_mode_all;
            }
            
            //void update_listing_mode();
        };

}

#endif /* PICVIZ_PVSTATEMACHINE_H */
