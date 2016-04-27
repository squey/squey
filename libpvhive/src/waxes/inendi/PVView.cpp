/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvhive/PVHive.h>
#include <pvhive/waxes/inendi/PVView.h>

PVHIVE_CALL_OBJECT_BLOCK_BEGIN()

// Processing waxes
//

IMPL_WAX(Inendi::PVView::process_eventline, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::process_eventline)>(view, args);
	refresh_observers(&view->get_output_layer());
}

IMPL_WAX(Inendi::PVView::process_selection, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::process_eventline)>(view, args);
	// refresh_observers(&view->get_pre_filter_layer());
}

IMPL_WAX(Inendi::PVView::process_layer_stack, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::process_layer_stack)>(view, args);
	refresh_observers(&view->get_layer_stack_output_layer());
}

IMPL_WAX(Inendi::PVView::process_visibility, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::process_visibility)>(view, args);
	// refresh_observers(&view->get_real_output_selection());
	refresh_observers(&view->get_output_layer());
}

IMPL_WAX(Inendi::PVView::process_from_selection, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::process_from_selection)>(view, args);
	refresh_observers(&view->get_post_filter_layer());
	refresh_observers(&view->get_output_layer());
}

IMPL_WAX(Inendi::PVView::process_from_layer_stack, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::process_from_layer_stack)>(view, args);
	refresh_observers(&view->get_layer_stack_output_layer());
	refresh_observers(&view->get_post_filter_layer());
	refresh_observers(&view->get_output_layer());
}

IMPL_WAX(Inendi::PVView::process_from_eventline, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::process_from_eventline)>(view, args);
	refresh_observers(&view->get_real_output_selection());
	refresh_observers(&view->get_output_layer());
}

IMPL_WAX(Inendi::PVView::process_real_output_selection, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::process_real_output_selection)>(view,
	                                                                                         args);
	refresh_observers(&view->get_real_output_selection());
}

// Layer stack waxes
//

IMPL_WAX(Inendi::PVView::add_new_layer, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::add_new_layer)>(view, args);
	refresh_observers(&view->get_layer_stack());
	refresh_observers(&view->get_layer_stack().get_selected_layer_index());
}

IMPL_WAX(Inendi::PVView::add_new_layer_from_file, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::add_new_layer_from_file)>(view, args);
	refresh_observers(&view->get_layer_stack());
	refresh_observers(&view->get_layer_stack().get_selected_layer_index());
}

IMPL_WAX(Inendi::PVView::delete_layer_n, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::delete_layer_n)>(view, args);
	refresh_observers(&view->get_layer_stack());
	refresh_observers(&view->get_layer_stack().get_selected_layer_index());
}

IMPL_WAX(Inendi::PVView::delete_selected_layer, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::delete_selected_layer)>(view, args);
	refresh_observers(&view->get_layer_stack());
	refresh_observers(&view->get_layer_stack().get_selected_layer_index());
}

IMPL_WAX(Inendi::PVView::move_selected_layer_to, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::move_selected_layer_to)>(view, args);
	refresh_observers(&view->get_layer_stack());
	refresh_observers(&view->get_layer_stack().get_selected_layer_index());
}

IMPL_WAX(Inendi::PVView::duplicate_selected_layer, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::duplicate_selected_layer)>(view, args);
	refresh_observers(&view->get_layer_stack());
	refresh_observers(&view->get_layer_stack().get_selected_layer_index());
}

IMPL_WAX(Inendi::PVView::reset_layers, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::reset_layers)>(view, args);
	refresh_observers(&view->get_layer_stack());
}

IMPL_WAX(Inendi::PVView::set_layer_stack_selected_layer_index, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::set_layer_stack_selected_layer_index)>(
	    view, args);
	refresh_observers(&view->get_layer_stack());
	refresh_observers(&view->get_layer_stack().get_selected_layer_index());
}

IMPL_WAX(Inendi::PVView::set_layer_stack_layer_n_name, view, args)
{
	auto ret =
	    call_object_default<Inendi::PVView, FUNC(Inendi::PVView::set_layer_stack_layer_n_name)>(
	        view, args);
	refresh_observers(&view->get_layer_stack());
	return ret;
}

IMPL_WAX(Inendi::PVView::toggle_layer_stack_layer_n_locked_state, view, args)
{
	auto ret = call_object_default<
	    Inendi::PVView, FUNC(Inendi::PVView::toggle_layer_stack_layer_n_locked_state)>(view, args);
	refresh_observers(&view->get_layer_stack());
	return ret;
}

IMPL_WAX(Inendi::PVView::toggle_layer_stack_layer_n_visible_state, view, args)
{
	auto ret = call_object_default<
	    Inendi::PVView, FUNC(Inendi::PVView::toggle_layer_stack_layer_n_visible_state)>(view, args);
	refresh_observers(&view->get_layer_stack());
	return ret;
}

IMPL_WAX(Inendi::PVView::toggle_view_unselected_zombie_visibility, view, args)
{
	call_object_default<Inendi::PVView,
	                    FUNC(Inendi::PVView::toggle_view_unselected_zombie_visibility)>(view, args);
	refresh_observers(&view->are_view_unselected_zombie_visible());
}

IMPL_WAX(Inendi::PVView::hide_layers, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::hide_layers)>(view, args);
	refresh_observers(&view->get_layer_stack());
}

IMPL_WAX(Inendi::PVView::compute_layer_min_max, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::compute_layer_min_max)>(view, args);
	refresh_observers(&view->get_layer_stack().get_selected_layer_index());
}

// Axes combination waxes
//

IMPL_WAX(Inendi::PVView::set_axes_combination_list_id, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::set_axes_combination_list_id)>(view,
	                                                                                        args);
	refresh_observers(&view->get_axes_combination().get_axes_index_list());
}

IMPL_WAX(Inendi::PVView::move_axis_to_new_position, view, args)
{
	auto ret = call_object_default<Inendi::PVView, FUNC(Inendi::PVView::move_axis_to_new_position)>(
	    view, args);
	refresh_observers(&view->get_axes_combination().get_axes_index_list());
	return ret;
}

IMPL_WAX(Inendi::PVView::remove_column, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::remove_column)>(view, args);
	refresh_observers(&view->get_axes_combination().get_axes_index_list());
}

IMPL_WAX(Inendi::PVView::axis_append, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::axis_append)>(view, args);
	refresh_observers(&view->get_axes_combination().get_axes_index_list());
}

PVHIVE_CALL_OBJECT_BLOCK_END()
