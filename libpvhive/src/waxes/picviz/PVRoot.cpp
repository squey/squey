/**
 * \file PVRoot.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvhive/PVHive.h>
#include <pvhive/waxes/picviz/PVRoot.h>

#include <picviz/PVRoot.h>
#include <picviz/PVView.h>

#include <iostream>

PVHIVE_CALL_OBJECT_BLOCK_BEGIN()

IMPL_WAX(Picviz::PVRoot::select_view, root, args)
{
	Picviz::PVView* old_cur_view = root->current_view();
	call_object_default<Picviz::PVRoot, FUNC(Picviz::PVRoot::select_view)>(root, args);
	if (old_cur_view) {
		refresh_observers(old_cur_view);
	}
	refresh_observers(&std::get<0>(args));
}

IMPL_WAX(Picviz::PVRoot::add_correlation, root, args)
{
	call_object_default<Picviz::PVRoot, FUNC(Picviz::PVRoot::add_correlation)>(root, args);
	refresh_observers(&root->get_correlations());
}

IMPL_WAX(Picviz::PVRoot::delete_correlation, root, args)
{
	call_object_default<Picviz::PVRoot, FUNC(Picviz::PVRoot::delete_correlation)>(root, args);
	refresh_observers(&root->get_correlations());
}

IMPL_WAX(Picviz::PVRoot::process_correlation, root, args)
{
	QList<Picviz::PVView*> changed_views = call_object_default<Picviz::PVRoot, FUNC(Picviz::PVRoot::process_correlation)>(root, args);
	refresh_observers(&root->get_correlations());
	for (Picviz::PVView* view : changed_views) {
		refresh_observers(&view->get_pre_filter_layer());
		refresh_observers(&view->get_post_filter_layer());
		refresh_observers(&view->get_output_layer());
		refresh_observers(&view->get_real_output_selection());
	}
	return changed_views;
}

PVHIVE_CALL_OBJECT_BLOCK_END()
