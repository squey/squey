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
		refresh_observers(&v->get_post_filter_layer());
		refresh_observers(&v->get_output_layer());
	}
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

PVHIVE_CALL_OBJECT_BLOCK_END()
