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

IMPL_WAX(Inendi::PVView::process_visibility, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::process_visibility)>(view, args);
	// refresh_observers(&view->get_real_output_selection());
	refresh_observers(&view->get_output_layer());
}

IMPL_WAX(Inendi::PVView::process_from_selection, view, args)
{
	Inendi::PVView* v =
	    call_object_default<Inendi::PVView, FUNC(Inendi::PVView::process_from_selection)>(view,
	                                                                                      args);
	if (v) {
		refresh_observers(&v->get_post_filter_layer());
		refresh_observers(&v->get_output_layer());
	}
	refresh_observers(&view->get_post_filter_layer());
	refresh_observers(&view->get_output_layer());

	return v;
}

IMPL_WAX(Inendi::PVView::process_from_layer_stack, view, args)
{
	Inendi::PVView* v =
	    call_object_default<Inendi::PVView, FUNC(Inendi::PVView::process_from_layer_stack)>(view,
	                                                                                        args);
	if (v) {
		refresh_observers(&v->get_layer_stack_output_layer());
		refresh_observers(&v->get_post_filter_layer());
		refresh_observers(&v->get_output_layer());
	}
	refresh_observers(&view->get_layer_stack_output_layer());
	refresh_observers(&view->get_post_filter_layer());
	refresh_observers(&view->get_output_layer());

	return v;
}

IMPL_WAX(Inendi::PVView::process_from_eventline, view, args)
{
	Inendi::PVView* v =
	    call_object_default<Inendi::PVView, FUNC(Inendi::PVView::process_from_eventline)>(view,
	                                                                                      args);

	refresh_observers(&view->get_real_output_selection());
	refresh_observers(&view->get_output_layer());

	if (v) {
		refresh_observers(&v->get_real_output_selection());
		refresh_observers(&v->get_output_layer());
	}

	return v;
}

IMPL_WAX(Inendi::PVView::process_real_output_selection, view, args)
{
	Inendi::PVView* v =
	    call_object_default<Inendi::PVView, FUNC(Inendi::PVView::process_real_output_selection)>(
	        view, args);
	refresh_observers(&view->get_real_output_selection());

	if (v) {
		refresh_observers(&v->get_real_output_selection());
	}

	return v;
}

// Layer stack waxes
//

IMPL_WAX(Inendi::PVView::add_new_layer, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::add_new_layer)>(view, args);
	refresh_observers(&view->get_layer_stack());
}

IMPL_WAX(Inendi::PVView::delete_layer_n, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::delete_layer_n)>(view, args);
	refresh_observers(&view->get_layer_stack());
}

IMPL_WAX(Inendi::PVView::delete_selected_layer, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::delete_selected_layer)>(view, args);
	refresh_observers(&view->get_layer_stack());
}

IMPL_WAX(Inendi::PVView::move_selected_layer_to, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::move_selected_layer_to)>(view, args);
	refresh_observers(&view->get_layer_stack());
}

IMPL_WAX(Inendi::PVView::duplicate_selected_layer, view, args)
{
	call_object_default<Inendi::PVView, FUNC(Inendi::PVView::duplicate_selected_layer)>(view, args);
	refresh_observers(&view->get_layer_stack());
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
}

IMPL_WAX(Inendi::PVView::set_layer_stack_layer_n_name, view, args)
{
	auto ret =
	    call_object_default<Inendi::PVView, FUNC(Inendi::PVView::set_layer_stack_layer_n_name)>(
	        view, args);
	refresh_observers(&view->get_layer_stack());
	return ret;
}

IMPL_WAX(Inendi::PVView::toggle_layer_stack_layer_n_visible_state, view, args)
{
	auto ret = call_object_default<Inendi::PVView,
	                               FUNC(Inendi::PVView::toggle_layer_stack_layer_n_visible_state)>(
	    view, args);
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

PVHIVE_CALL_OBJECT_BLOCK_END()
