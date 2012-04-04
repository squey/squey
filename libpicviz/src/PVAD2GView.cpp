//! \file PVAD2GView.cpp
//! Copyright (C) Picviz Labs 2012

#include <picviz/PVAD2GView.h>
#include <picviz/PVAD2GViewValueContainer.h>
#include <picviz/PVCombiningFunctionView.h>
#include <picviz/PVView.h>

#include <tulip/Graph.h>
#include <tulip/Node.h>
#include <tulip/PropertyTypes.h>

#include <set>
#include <queue>

/******************************************************************************
 *
 * Classes for Tulip graph property
 *
 *****************************************************************************/

namespace Picviz {

typedef PVAD2GViewValueContainer<Picviz::PVView*> PVAD2GViewNode;

class PVAD2GViewNodeType : public tlp::TypeInterface <Picviz::PVAD2GViewNode> {
public:
	static std::string toString(const RealType &value) {
		// TODO: write serialization exporter
		return "";
	}
	static bool fromString(RealType &value, const std::string &str){
		// TODO: write serialization importer
		return true;
	}
};

typedef PVAD2GViewValueContainer<PVCombiningFunctionView_p> PVAD2GViewEdge;

class PVAD2GViewEdgeType : public tlp::TypeInterface <Picviz::PVAD2GViewEdge> {
public:
	static std::string toString(const RealType &value) {
		// TODO: write serialization exporter
		return "";
	}
	static bool fromString(RealType &value, const std::string &str){
		// TODO: write serialization importer
		return true;
	}
};

typedef tlp::AbstractProperty<Picviz::PVAD2GViewNodeType /* Tnode */,
                              Picviz::PVAD2GViewEdgeType /* Tedge */,
                              tlp::Algorithm> AbstractPVAD2GViewCorrelationProperty;

class PVAD2GViewCorrelationAlgorithm : public tlp::Algorithm {};

class PVAD2GViewCorrelationProperty : public AbstractPVAD2GViewCorrelationProperty {
	friend class PVAD2GViewCorrelationAlgorithm;

public :
	PVAD2GViewCorrelationProperty (tlp::Graph *g, std::string n="") :
		AbstractPVAD2GViewCorrelationProperty(g, n) {}

	// PropertyInterface inherited methods
	tlp::PropertyInterface* clonePrototype(tlp::Graph *g, const std::string& n) {
		if( !g )
			return 0;

		// allow to get an unregistered property (empty name)
		PVAD2GViewCorrelationProperty * p = n.empty()
			? new PVAD2GViewCorrelationProperty(g) : g->getLocalProperty<PVAD2GViewCorrelationProperty>(n);
		p->setAllNodeValue(getNodeDefaultValue());
		p->setAllEdgeValue(getEdgeDefaultValue());
		return p;
	}

	static const std::string propertyTypename;
	std::string getTypename() const { return propertyTypename; }
};

const std::string PVAD2GViewCorrelationProperty::propertyTypename = "PVAD2GViewCorrelationProperty";
}

/******************************************************************************
 *
 * temporary code
 *
 *****************************************************************************/

tlp::Graph *create_graph()
{
	tlp::Graph *graph = tlp::newGraph();

	tlp::node va = graph->addNode();
	tlp::node vb = graph->addNode();
	tlp::node vc = graph->addNode();
	tlp::node vd = graph->addNode();

	tlp::edge f0 = graph->addEdge(va, vb);
	tlp::edge f1 = graph->addEdge(va, vc);
	tlp::edge f2 = graph->addEdge(vb, vd);
	tlp::edge f3 = graph->addEdge(vd, vb);

	(void)f0;
	(void)f1;
	(void)f2;
	(void)f3;

	return graph;
}

int count_paths_num(tlp::Graph *graph, tlp::node a, tlp::node b);

/******************************************************************************
 *
 * Picviz::PVAD2GView::PVAD2GView
 *
 *****************************************************************************/
Picviz::PVAD2GView::PVAD2GView()
{
	_graph = create_graph();
	_corr_info = _graph->getLocalProperty<PVAD2GViewCorrelationProperty>("correlationProperty");
}

/******************************************************************************
 *
 * Picviz::PVAD2GView::~PVAD2GView
 *
 *****************************************************************************/
Picviz::PVAD2GView::~PVAD2GView()
{
	if(_graph != 0)
		delete _graph;
}

/******************************************************************************
 *
 * Picviz::PVAD2GView::run
 *
 *****************************************************************************/
void Picviz::PVAD2GView::run(Picviz::PVView *view)
{
	/**
	 * This method uses an iterative breadth-first graph traversal to
	 * propagate the selection update to every PVView.
	 * The "redraw" is done by the caller.
	 */
	tlp::node node, next;
	tlp::edge edge;
	std::set<tlp::node> visited;
	std::queue<tlp::node> pending;
	Picviz::PVCombiningFunctionView_p cfview;
	Picviz::PVSelection selection;
	Picviz::PVView *va, *vb;

	node = get_graph_node(view);

	if(node.isValid() == false)
		return;

	pending.push(node);

	while(pending.size()) {
		node = pending.front();
		pending.pop();
		va = _corr_info->getNodeValue(node).get_data();

		forEach(edge, _graph->getOutEdges(node)) {
			next = _graph->target(edge);

			// a PVView is only updated once
			if (visited.find(next) != visited.end())
				continue;

			vb = _corr_info->getNodeValue(next).get_data();
			cfview = _corr_info->getEdgeValue(edge).get_data();

			selection = (*cfview)(*va, *vb);
			vb->set_selection_view(selection);

			pending.push(next);
			visited.insert(next);
		}
	}
}

/******************************************************************************
 *
 * Picviz::PVAD2GView::check_properties
 *
 *****************************************************************************/
bool Picviz::PVAD2GView::check_properties()
{
	return true;
}

int count_paths_num(tlp::Graph *graph, tlp::node na, tlp::node nb)
{
	int count = 0;

	return count;
}


/******************************************************************************
 *
 * Picviz::PVAD2GView::add_node
 *
 *****************************************************************************/
bool Picviz::PVAD2GView::add_node(Picviz::PVView *view)
{
	tlp::node node;

	node = get_graph_node(view);

	if(node.isValid() == true)
		return false;

	_corr_info->setNodeValue(node, view);

	return true;
}

/******************************************************************************
 *
 * Picviz::PVAD2GView::set_edge_f
 *
 *****************************************************************************/
bool Picviz::PVAD2GView::set_edge_f(const Picviz::PVView *va, const Picviz::PVView *vb,
                                    PVCombiningFunctionView_p cfview)
{
	tlp::node na, nb;
	tlp::edge e;

	na = get_graph_node(va);
	nb = get_graph_node(vb);

	if((na.isValid() == false) || (nb.isValid() == false))
		return false;

	e = _graph->existEdge(na, nb, true);

	if(e.isValid() == false) {
		e = _graph->addEdge(na, nb);
		if(e.isValid() == false)
			return false;
	}

	_corr_info->getEdgeValue(e).set_data(cfview);

	return true;
}

/******************************************************************************
 *
 * Picviz::PVAD2GView::retrieve_graph_node
 *
 *****************************************************************************/
tlp::node Picviz::PVAD2GView::get_graph_node(const Picviz::PVView *view)
{
	tlp::node result; // a tlp::node is initialized to an invalid value
	tlp::node node;

	if(_graph != 0)
		return result;

	forEach(node, _graph->getNodes()) {
		if(view == _corr_info->getNodeValue(node).get_data())
			result = node;
	}

	return result;
}
