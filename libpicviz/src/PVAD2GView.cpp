//! \file PVAD2GView.cpp
//! Copyright (C) Picviz Labs 2012

#include <picviz/PVAD2GView.h>
#include <picviz/PVSimpleContainerTmpl.h>
#include <picviz/PVCombiningFunctionView.h>
#include <picviz/PVView.h>

#include <tulip/Graph.h>
#include <tulip/Node.h>
#include <tulip/PropertyTypes.h>

#include <queue>

// some macro to make things clearer
// Tulip nodes and edges are implicitly initialized to an invalid value
#define TLP_NODE_INVALID tlp::node()
#define TLP_EDGE_INVALID tlp::edge()

/******************************************************************************
 *
 * Classes for Tulip graph property
 *
 *****************************************************************************/

/* To add a new property to a Tulip graph, some classes are needed:
 * - a pointer can not be used as a graph property, so it must be encapsulated:
 *   PVAD2GViewNode and PVAD2GViewEdge
 * - an interface is needed to (de)serialize the property's values:
 *   PVAD2GViewNodeType and PVAD2GViewEdgeType
 * - an algorithm: PVAD2GViewCorrelationAlgorithm
 * - finally the property definition: AbstractPVAD2GViewCorrelationProperty and
 *   PVAD2GViewCorrelationProperty
 */
namespace Picviz {

typedef PVSimpleContainerTmpl<Picviz::PVView*> PVAD2GViewNode;

class PVAD2GViewNodeType : public tlp::TypeInterface <Picviz::PVAD2GViewNode> {
public:
	static std::string toString(const RealType &/*value*/) {
		// TODO: write serialization exporter
		return "";
	}
	static bool fromString(RealType &/*value*/, const std::string &/*str*/){
		// TODO: write serialization importer
		return true;
	}
};

typedef PVSimpleContainerTmpl<PVCombiningFunctionView_p> PVAD2GViewEdge;

class PVAD2GViewEdgeType : public tlp::TypeInterface <Picviz::PVAD2GViewEdge> {
public:
	static std::string toString(const RealType &/*value*/) {
		// TODO: write serialization exporter
		return "";
	}
	static bool fromString(RealType &/*value*/, const std::string &/*str*/){
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
		if( !g ) {
			return 0;
		}

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
 * Picviz::PVAD2GView::PVAD2GView
 *
 *****************************************************************************/
Picviz::PVAD2GView::PVAD2GView(Picviz::PVScene* scene) :
	_scene(scene)
{
	_graph = tlp::newGraph();
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
 * Picviz::PVAD2GView::add_view
 *
 *****************************************************************************/
tlp::node Picviz::PVAD2GView::add_view(Picviz::PVView *view)
{
	tlp::node node;

	node = get_graph_node(view);

	if(node.isValid() == true)
		return node;

	node = _graph->addNode();
	if(node.isValid() == false)
		return TLP_NODE_INVALID;

	_corr_info->setNodeValue(node, view);

	return node;
}

/******************************************************************************
 *
 * Picviz::PVAD2GView::add_view
 *
 *****************************************************************************/
Picviz::PVView* Picviz::PVAD2GView::get_view(tlp::node n)
{
	if(_graph->isElement(n) == false)
		return NULL;

	return _corr_info->getNodeValue(n).get_data();
}

/******************************************************************************
 *
 * Picviz::PVAD2GView::set_edge_f
 *
 *****************************************************************************/
tlp::edge Picviz::PVAD2GView::set_edge_f(const Picviz::PVView *va,
                                         const Picviz::PVView *vb,
                                         PVCombiningFunctionView_p cfview)
{
	tlp::node na, nb;
	tlp::edge edge;

	na = get_graph_node(va);
	nb = get_graph_node(vb);

	if((na.isValid() == false) || (nb.isValid() == false))
		return TLP_EDGE_INVALID;

	edge = _graph->existEdge(na, nb, true);

	if(edge.isValid() == false) {
		edge = _graph->addEdge(na, nb);
		if(edge.isValid() == false)
			return TLP_EDGE_INVALID;
	}

	_corr_info->setEdgeValue(edge, cfview);

	return edge;
}

/******************************************************************************
 *
 * Picviz::PVAD2GView::get_edge_f
 *
 *****************************************************************************/
Picviz::PVCombiningFunctionView_p Picviz::PVAD2GView::get_edge_f(const tlp::edge edge) const
{
	// an invalid value when the edge is not in the Tulip graph
	if(_graph->isElement(edge) == false)
		return Picviz::PVCombiningFunctionView_p();


	return _corr_info->getEdgeValue(edge).get_data();
}

/******************************************************************************
 *
 * Picviz::PVAD2GView::visit_from_view_f
 *
 *****************************************************************************/
void Picviz::PVAD2GView::visit_edges_f(graph_func_t const& f) const
{
	tlp::edge edge;
	tlp::node a, b;

	forEach(edge, _graph->getEdges()) {
		a = _graph->source(edge);
		b = _graph->target(edge);
		f(*_corr_info->getEdgeValue(edge).get_data(),
		  *_corr_info->getNodeValue(a).get_data(),
		  *_corr_info->getNodeValue(b).get_data());
	}
}

/******************************************************************************
 *
 * Picviz::PVAD2GView::visit_from_view_f
 *
 *****************************************************************************/
void Picviz::PVAD2GView::visit_from_view_f(Picviz::PVView *view, graph_func_t const& f) const
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
	Picviz::PVCombiningFunctionView_p cfview_p;
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
			PVLOG_INFO("propagating from view %p to view %p\n",
			            va, vb);
			cfview_p = _corr_info->getEdgeValue(edge).get_data();

			f(*cfview_p, *va, *vb);

			pending.push(next);
			visited.insert(next);
		}
		visited.insert(node);
	}
}

/******************************************************************************
 *
 * Picviz::PVAD2GView::check_properties
 *
 *****************************************************************************/
bool Picviz::PVAD2GView::check_properties()
{
	tlp::node a, b;
	forEach(a, _graph->getNodes()) {
		forEach(b, _graph->getNodes()) {
			if(count_path_number(a, b) > 1) {
				returnForEach(false);
			}
		}
	}

	return true;
}

/******************************************************************************
 *
 * Picviz::PVAD2GView::retrieve_graph_node
 *
 *****************************************************************************/
tlp::node Picviz::PVAD2GView::get_graph_node(const Picviz::PVView *view) const
{
	tlp::node node;

	if(_graph == 0)
		return TLP_NODE_INVALID;

	forEach(node, _graph->getNodes()) {
		if(view == _corr_info->getNodeValue(node).get_data())
			returnForEach(node);
	}

	return TLP_NODE_INVALID;
}

/******************************************************************************
 *
 * Picviz::PVAD2GView::count_path_number
 *
 *****************************************************************************/
int Picviz::PVAD2GView::count_path_number(const tlp::node& a, const tlp::node& b) const
{
	int count = 0;
	graph_path_t path;
	graph_visited_t visited;

	count_path_number_rec(a, b, count, path, visited);

	return count;
}

/******************************************************************************
 *
 * Picviz::PVAD2GView::count_path_number_rec
 *
 *****************************************************************************/
void Picviz::PVAD2GView::count_path_number_rec(const tlp::node& a, const tlp::node& b, int& count, graph_path_t& path, graph_visited_t& visited) const
{
	tlp::node next;

	if (visited.find(a) != visited.end())
		return;

	path.push_back(a);
	visited.insert(a);

	forEach(next, _graph->getOutNodes(a)) {
		bool loop_in_path = false;

		// skipping paths with loop
		for(graph_path_t::iterator it = path.begin(); it != path.end(); ++it)
			if(*it == next)
				loop_in_path = true;
		if(loop_in_path)
			continue;


		if(next == b) {
			++count;
			// for(node_list::iterator it = path.begin(); it != path.end(); ++it)
			// 	std::cout << *it << ", ";
			// std::cout << next.id << std::endl;
		}

		count_path_number_rec(next, b, count, path, visited);
	}

	path.pop_back(); // remove node from path
}
