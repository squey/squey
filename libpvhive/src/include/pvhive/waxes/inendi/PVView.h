/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVHIVE_WAXES_INENDI_PVVIEW_H
#define PVHIVE_WAXES_INENDI_PVVIEW_H

#include <pvhive/PVWax.h>
#include <inendi/PVView.h>

// Processing waxes
//

DECLARE_WAX(Inendi::PVView::process_eventline)
DECLARE_WAX(Inendi::PVView::process_selection)
DECLARE_WAX(Inendi::PVView::process_layer_stack)
DECLARE_WAX(Inendi::PVView::process_visibility)
DECLARE_WAX(Inendi::PVView::process_from_selection)
DECLARE_WAX(Inendi::PVView::process_from_layer_stack)
DECLARE_WAX(Inendi::PVView::process_from_eventline)
DECLARE_WAX(Inendi::PVView::process_real_output_selection)

// Layer stack waxes
//

DECLARE_WAX(Inendi::PVView::add_new_layer)
DECLARE_WAX(Inendi::PVView::delete_layer_n)
DECLARE_WAX(Inendi::PVView::delete_selected_layer)
DECLARE_WAX(Inendi::PVView::move_selected_layer_to)
DECLARE_WAX(Inendi::PVView::reset_layers)
DECLARE_WAX(Inendi::PVView::set_layer_stack_selected_layer_index)
DECLARE_WAX(Inendi::PVView::set_layer_stack_layer_n_name)
DECLARE_WAX(Inendi::PVView::toggle_layer_stack_layer_n_locked_state)
DECLARE_WAX(Inendi::PVView::toggle_layer_stack_layer_n_visible_state)
DECLARE_WAX(Inendi::PVView::toggle_view_unselected_zombie_visibility)
DECLARE_WAX(Inendi::PVView::hide_layers)
DECLARE_WAX(Inendi::PVView::compute_layer_min_max)

// Axes combination waxes
//

DECLARE_WAX(Inendi::PVView::set_axes_combination_list_id)
DECLARE_WAX(Inendi::PVView::move_axis_to_new_position)
DECLARE_WAX(Inendi::PVView::remove_column)
DECLARE_WAX(Inendi::PVView::axis_append)

#endif
