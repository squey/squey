/**
 * \file PVAD2GView.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

IMPL_WAX(Picviz::PVAD2GView::add_view, ad2gview, args)
{
	Picviz::PVAD2GView* ret = call_object_default<Picviz::PVAD2GView, FUNC(Picviz::PVAD2GView::add_view)>(ad2gview, args);
	refresh_observers(&(args->get_parent<Picviz::PVRoot>()->get_correlations()));

	return ret;
}
