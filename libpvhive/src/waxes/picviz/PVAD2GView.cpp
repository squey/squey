/**
 * \file PVAD2GView.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <tulip/Node.h>

#include <pvhive/PVHive.h>
#include <pvhive/waxes/picviz/PVAD2GView.h>

#include <picviz/PVRoot.h>
#include <picviz/PVView.h>

PVHIVE_CALL_OBJECT_BLOCK_BEGIN()

IMPL_WAX(Picviz::PVAD2GView::add_view, ad2gview, args)
{
	Picviz::PVView* view = std::get<0>(args);
	tlp::node ret = call_object_default<Picviz::PVAD2GView, FUNC(Picviz::PVAD2GView::add_view)>(ad2gview, view);
	Picviz::PVRoot* root = view->get_parent<Picviz::PVRoot>();
	refresh_observers(&root->get_correlations());

	return ret;
}

IMPL_WAX(Picviz::PVAD2GView::del_view_by_node, ad2gview, args)
{
	tlp::node n = std::get<0>(args);
	Picviz::PVView* view = ad2gview->get_view(n);
	call_object_default<Picviz::PVAD2GView, FUNC(Picviz::PVAD2GView::del_view_by_node)>(ad2gview, args);
	Picviz::PVRoot* root = view->get_parent<Picviz::PVRoot>();
	refresh_observers(&root->get_correlations());
}

/*IMPL_WAX(Picviz::PVAD2GView::set_edge_f, ad2gview, args)
{
	tlp::node source = std::get<0>(args);
	//tlp::node dest = std::get<1>(args);
	Picviz::PVView* source_view = ad2gview->get_view(source);
	//Picviz::PVView* dest_view = ad2gview->get_view(dest);
	call_object_default<Picviz::PVAD2GView, FUNC(Picviz::PVAD2GView::set_edge_f)>(ad2gview, args);
	Picviz::PVRoot* root = source_view->get_parent<Picviz::PVRoot>();
	refresh_observers(&root->get_correlations());
}*/

PVHIVE_CALL_OBJECT_BLOCK_END()
