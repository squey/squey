//! \file PVAD2GView.cpp
//! Copyright (C) Picviz Labs 2012

#include <picviz/PVAD2GView.h>
#include <picviz/PVAD2GViewPointerContainer.h>

#include <picviz/PVView.h>

#include <tulip/Graph.h>
#include <tulip/Node.h>
#include <tulip/PropertyTypes.h>

/******************************************************************************
 *
 * Classes for Tulip graph property
 *
 *****************************************************************************/

namespace Picviz {

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
Picviz::PVAD2GView::PVAD2GView():
	_graph(0)
{
	_graph = create_graph();
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
	/* if a node has view as pointer, return false
	   else add new node and set view
	 */
}

/******************************************************************************
 *
 * Picviz::PVAD2GView::set_edge_f
 *
 *****************************************************************************/
void Picviz::PVAD2GView::set_edge_f(Picviz::PVView *va, Picviz::PVView *vb,
                                    PVCombiningFunctionView *cfview)
{
	// TODO
}
