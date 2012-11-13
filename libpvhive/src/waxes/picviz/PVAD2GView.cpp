/**
 * \file PVAD2GView.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <tulip/Node.h>
#include <tulip/Edge.h>
#include <tulip/Graph.h>

#include <pvhive/PVHive.h>
#include <pvhive/waxes/picviz/PVAD2GView.h>

#include <picviz/PVRoot.h>
#include <picviz/PVView.h>
#include <picviz/PVCombiningFunctionView.h>

PVHIVE_CALL_OBJECT_BLOCK_BEGIN()

IMPL_WAX(Picviz::PVAD2GView::del_view_by_node, ad2gview, args)
{
	tlp::node n = std::get<0>(args);
	Picviz::PVView* view = ad2gview->get_view(n);
	call_object_default<Picviz::PVAD2GView, FUNC(Picviz::PVAD2GView::del_view_by_node)>(ad2gview, args);
	Picviz::PVRoot* root = view->get_parent<Picviz::PVRoot>();
	refresh_observers(&root->get_correlations());
}

IMPL_WAX(Picviz::PVAD2GView::del_edge, ad2gview, args)
{
	tlp::edge e = std::get<0>(args);
	tlp::node source_node = ad2gview->get_graph()->source(e);
	tlp::node target_node = ad2gview->get_graph()->target(e);
	Picviz::PVView* source_view = ad2gview->get_view(source_node);
	Picviz::PVView* target_view = ad2gview->get_view(target_node);
	call_object_default<Picviz::PVAD2GView, FUNC(Picviz::PVAD2GView::del_edge)>(ad2gview, args);
	Picviz::PVRoot* source_root = source_view->get_parent<Picviz::PVRoot>();
	Picviz::PVRoot* target_root = target_view->get_parent<Picviz::PVRoot>();
	refresh_observers(&source_root->get_correlations());
	if (target_root != source_root) {
		refresh_observers(&target_root->get_correlations());
	}
}

IMPL_WAX(Picviz::PVAD2GView::set_edge_by_node_f, ad2gview, args)
{
	tlp::node source_node = std::get<0>(args);
	tlp::node target_node = std::get<1>(args);
	Picviz::PVView* source_view = ad2gview->get_view(source_node);
	Picviz::PVView* target_view = ad2gview->get_view(target_node);
	tlp::edge edge = call_object_default<Picviz::PVAD2GView, FUNC(Picviz::PVAD2GView::set_edge_by_node_f)>(ad2gview, args);
	Picviz::PVRoot* source_root = source_view->get_parent<Picviz::PVRoot>();
	Picviz::PVRoot* target_root = target_view->get_parent<Picviz::PVRoot>();
	refresh_observers(&source_root->get_correlations());
	if (target_root != source_root) {
		refresh_observers(&target_root->get_correlations());
	}

	return edge;
}

PVHIVE_CALL_OBJECT_BLOCK_END()
