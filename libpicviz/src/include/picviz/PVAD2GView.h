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
 * \class PVLayerFilter
 */
class PVAD2GView
{
public:
	PVAD2GView();
	~PVAD2GView();

public:
	tlp::Graph *get_graph() { return _graph; }
	void set_graph(tlp::Graph *graph);

public:
	void run(Picviz::PVView *view);

	bool check_properties();

	tlp::node add_view(Picviz::PVView *view);

	tlp::edge set_edge_f(const Picviz::PVView *va, const Picviz::PVView *vb,
	                     PVCombiningFunctionView_p cfview);

private:
	tlp::node get_graph_node(const Picviz::PVView *view);

private:
	/* graph tulip object */
	tlp::Graph *_graph;
	/* graph's property */
	PVAD2GViewCorrelationProperty *_corr_info;
};

}

#endif // PICVIZ_PVAD2GVIEW_H
