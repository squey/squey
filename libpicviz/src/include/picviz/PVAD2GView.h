#ifndef PICVIZ_PVAD2GVIEW_H
#define PICVIZ_PVAD2GVIEW_H

#include <pvkernel/core/general.h>
#include <picviz/PVView.h>
#include <picviz/PVCombiningFunctionView_types.h>

// forward declaration of tlp::Graph and tlp::node
namespace tlp {
class Graph;
struct node;
}

namespace Picviz {

// forward declaration of the class for multi-source correlation property
class PVAD2GViewCorrelationProperty;

/**
 * \class PVAD2GView
 */
class PVAD2GView
{
public:
	PVAD2GView();
	~PVAD2GView();

	tlp::Graph *get_graph() { return _graph; }

	void run(Picviz::PVView *view);

	bool check_properties();

	bool add_node(Picviz::PVView *view);

	bool set_edge_f(const Picviz::PVView *va, const Picviz::PVView *vb,
	                PVCombiningFunctionView_p cfview);

private:
	tlp::node retrieve_graph_node(const Picviz::PVView *view);

private:
	/* graph tulip object */
	tlp::Graph *_graph;
	/* graph's property */
	PVAD2GViewCorrelationProperty *_corr_info;
};

}

#endif // PICVIZ_PVAD2GVIEW_H
