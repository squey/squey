#ifndef PICVIZ_PVAD2GVIEW_H
#define PICVIZ_PVAD2GVIEW_H

#include <pvkernel/core/general.h>
#include <picviz/PVView.h>
#include <picviz/PVCombiningFunctionView.h>

// forward declaration of tlp::Graph and tlp::node
namespace tlp {
class Graph;
struct node;
}

namespace Picviz {

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

	void set_edge_f(Picviz::PVView *va, Picviz::PVView *vb,
	                PVCombiningFunctionView *cfview);

private:
	/* graph tulip object */
	tlp::Graph *_graph;
};

}

#endif // PICVIZ_PVAD2GVIEW_H
