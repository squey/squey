#ifndef PICVIZ_PVAD2GVIEW_H
#define PICVIZ_PVAD2GVIEW_H

#include <pvkernel/core/general.h>
#include <picviz/PVCombiningFunctionView_types.h>

// forward declaration of tlp::Graph and tlp::node
namespace tlp {
class Graph;
struct node;
struct edge;
}

namespace Picviz {

// forward declaration of the class for multi-source correlation property
class PVAD2GViewCorrelationProperty;
class PVView;

/**
 * \class PVAD2GView
 */
class PVAD2GView
{
public:
	PVAD2GView();
	~PVAD2GView();

public:
	tlp::Graph *get_graph() { return _graph; }

public:
	tlp::node add_view(Picviz::PVView *view);

	tlp::edge set_edge_f(const Picviz::PVView *va, const Picviz::PVView *vb,
	                     PVCombiningFunctionView_p cfview);

	Picviz::PVCombiningFunctionView_p get_edge_f(const tlp::edge edge);

public:
	void run(Picviz::PVView *view);

	bool check_properties();

private:
	tlp::node get_graph_node(const Picviz::PVView *view);

	/* TODO: I need a method to check paths validity
	 *
	 * int count_paths_num(tlp::Graph *graph, tlp::node a, tlp::node b);
	 */

private:
	/* graph tulip object */
	tlp::Graph *_graph;
	/* graph's property */
	PVAD2GViewCorrelationProperty *_corr_info;
};

}

#endif // PICVIZ_PVAD2GVIEW_H
