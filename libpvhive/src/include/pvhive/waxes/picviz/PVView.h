/**
 * \file PVView.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef PVHIVE_WAXES_PICVIZ_PVVIEW_H
#define PVHIVE_WAXES_PICVIZ_PVVIEW_H

#include <pvhive/PVWax.h>
#include <picviz/PVView.h>

// Processing waxes
//

DECLARE_WAX(Picviz::PVView::process_eventline)
DECLARE_WAX(Picviz::PVView::process_selection)
DECLARE_WAX(Picviz::PVView::process_layer_stack)
DECLARE_WAX(Picviz::PVView::process_filter)
DECLARE_WAX(Picviz::PVView::process_visibility)
DECLARE_WAX(Picviz::PVView::process_from_selection)
DECLARE_WAX(Picviz::PVView::process_from_layer_stack)
DECLARE_WAX(Picviz::PVView::process_from_filter)
DECLARE_WAX(Picviz::PVView::process_from_eventline)
DECLARE_WAX(Picviz::PVView::process_real_output_selection)

// Layer stack waxes
//

DECLARE_WAX(Picviz::PVView::add_new_layer)
DECLARE_WAX(Picviz::PVView::add_new_layer_from_file)
DECLARE_WAX(Picviz::PVView::delete_layer_n)
DECLARE_WAX(Picviz::PVView::delete_selected_layer)
DECLARE_WAX(Picviz::PVView::commit_to_new_layer)
DECLARE_WAX(Picviz::PVView::reset_layers)
DECLARE_WAX(Picviz::PVView::set_layer_stack_selected_layer_index)
DECLARE_WAX(Picviz::PVView::set_layer_stack_layer_n_name)
DECLARE_WAX(Picviz::PVView::toggle_layer_stack_layer_n_locked_state)
DECLARE_WAX(Picviz::PVView::toggle_layer_stack_layer_n_visible_state)
DECLARE_WAX(Picviz::PVView::toggle_parallelview_unselected_zombie_visibility)

// Axes combination waxes
//

DECLARE_WAX(Picviz::PVView::set_axes_combination_list_id)
DECLARE_WAX(Picviz::PVView::move_axis_to_new_position)
DECLARE_WAX(Picviz::PVView::remove_column)
DECLARE_WAX(Picviz::PVView::axis_append)

#endif
