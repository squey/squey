//! \file PVAD2GView.cpp
//! Copyright (C) Picviz Labs 2012

#include <pvkernel/core/PVSerializeObject.h>
#include <pvkernel/core/PVSerializeArchive.h>

#include <picviz/PVAD2GView.h>
#include <picviz/PVSimpleContainerTmpl.h>
#include <picviz/PVCombiningFunctionView.h>
#include <picviz/PVView.h>

#include <tulip/Graph.h>
#include <tulip/Node.h>
#include <tulip/PropertyTypes.h>
#include <tulip/StringProperty.h>
#include <tulip/ColorProperty.h>

#include <sstream>
#include <queue>

// some macro to make things clearer
// Tulip nodes and edges are implicitly initialized to an invalid value
#define TLP_NODE_INVALID tlp::node()
#define TLP_EDGE_INVALID tlp::edge()

#define TLP_CORR_PROPERTY "correlation_property"
#define TLP_STR_CORR_PROPERTY "correlation_str_property"

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

// Tulip helper for stroing pointers and shared pointers

namespace tlp {

template <typename T>
struct StoredType<boost::shared_ptr<T> > {          
	typedef boost::shared_ptr<T> Value;
	typedef Value ReturnedValue;
	typedef Value const ReturnedConstValue;

	enum {isPointer=0};

	inline static ReturnedValue get(Value const& val) {
		return val;
	}

	inline static bool equal(const Value& val1, const Value& val2) {
		return val2 == val1;
	}

	inline static Value clone(const Value& val) {
		return val;
	}

	inline static void destroy(Value /*val*/) { }

	inline static Value defaultValue() {
		return Value();
	}
};

template <typename T>
struct StoredType<T*> {          
	typedef T* Value;
	typedef T* ReturnedValue;
	typedef T* ReturnedConstValue;

	enum {isPointer=0};

	inline static ReturnedValue get(Value val) {
		return val;
	}

	inline static bool equal(Value val1, Value val2) {
		return val2 == val1;
	}

	inline static Value clone(Value val) {
		return val;
	}

	inline static void destroy(Value /*val*/) { }

	inline static Value defaultValue() {
		return NULL;
	}
};

}

namespace Picviz {

class PVAD2GViewNodeType
{
public:
	typedef Picviz::PVView* RealType;

	static RealType undefinedValue() {
		return NULL;
	}
	static RealType defaultValue() {
		return NULL;
	}

	static void write(std::ostream&, const RealType&)
	{ }

	static bool read(std::istream&, RealType&)
	{
		return false;
	}

	static std::string toString(const RealType &value)
	{
		Picviz::PVView* view = value;
		// This can be called to know the string value of default value. So we must
		// handle the case where `view' is NULL.
		if (view == defaultValue()) {
			return std::string(); 
		}

		// Get weak pointer to the last serialized object of this view
		boost::weak_ptr<PVCore::PVSerializeObject> view_so = view->get_last_so();
		assert(!view_so.expired());
		return std::string(qPrintable(view_so.lock()->get_logical_path()));
	}
	static bool fromString(RealType& /*value*/, const std::string& /*str*/)
	{
		return false;
	}
};


typedef PVCombiningFunctionView_p PVAD2GViewEdge;

class PVAD2GViewEdgeType : public tlp::TypeInterface<PVCombiningFunctionView_p>
{
public:
	static std::string toString(const RealType &value) {
		std::string s;
		PVCombiningFunctionView_p cf = value;
		if (cf) {
			cf->to_string(s);
		}
		return s;
	}
	static bool fromString(RealType &value, const std::string &str){
		if (str.size() == 0) {
			value = PVCombiningFunctionView_p();
			return false;
		}
		PVCombiningFunctionView_p cf(new PVCombiningFunctionView());
		cf->from_string(str);
		value = cf;
		return true;
	}
};

typedef tlp::AbstractProperty<Picviz::PVAD2GViewNodeType /* Tnode */,
                              Picviz::PVAD2GViewEdgeType /* Tedge */,
                              tlp::Algorithm> AbstractPVAD2GViewCorrelationProperty;

class PVAD2GViewCorrelationProperty : public AbstractPVAD2GViewCorrelationProperty {
	friend class tlp::Algorithm;

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
	_corr_info = _graph->getLocalProperty<PVAD2GViewCorrelationProperty>(TLP_CORR_PROPERTY);
}

/******************************************************************************
 *
 * Picviz::PVAD2GView::~PVAD2GView
 *
 *****************************************************************************/
Picviz::PVAD2GView::~PVAD2GView()
{
	if (_graph) {
		delete _graph;
	}
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
 * Picviz::PVAD2GView::del_view
 *
 *****************************************************************************/
void Picviz::PVAD2GView::del_view(Picviz::PVView *view)
{
	tlp::node node;

	node = get_graph_node(view);

	if(node.isValid() == true)
		del_view_by_node(node);
}
/******************************************************************************
 *
 * Picviz::PVAD2GView::del_view_by_node
 *
 *****************************************************************************/
void Picviz::PVAD2GView::del_view_by_node(tlp::node node)
{
	tlp::edge edge;

	if(node.isValid() == false)
		return;

	// Tulip removes node and its connected edges but not the properties
	_corr_info->setNodeValue(node, 0);

	forEach(edge, _graph->getInOutEdges(node)) {
		_corr_info->setEdgeValue(edge, PVAD2GViewEdge());
	}

	_graph->delNode(node);
}

#if 0
tlp::Graph* Picviz::PVAD2GView::get_serializable_sub_graph() const
{
	// Get a sub graph of our graph that can be serialized (that is, whose views have been serialized).
	// A cleaner way to do this would be to set a BooleanProperty to our graph, whose values will be true
	// if and only if the corresponding nodes (views) are serializable.
	
	tlp::Graph* sub_graph = tlp::newGraph();
	PVAD2GViewCorrelationProperty* sub_corr_info = sub_graph->getLocalProperty<PVAD2GViewCorrelationProperty>(TLP_CORR_PROPERTY);
	tlp::copyToGraph(sub_graph, _graph);
	//sub_corr_info->copy(_corr_info);

	// Visit each node, and check whether the view has been serialized
	QList<tlp::node> nodes_to_del;
	tlp::node a;
	forEach(a, sub_graph->getNodes()) {
		Picviz::PVView const* view = sub_corr_info->getNodeValue(a);
		if (view->get_last_so().expired()) {
			// No serialisation has been made ! Remove this node from the graph.
			nodes_to_del.push_back(a);
		}
	}

	// Remove nodes from sub graph
	foreach(tlp::node node, nodes_to_del) {
		// Tulip removes node and its connected edges but not the properties
		sub_corr_info->setNodeValue(node, 0);

		tlp::edge edge;
		forEach(edge, sub_graph->getInOutEdges(node)) {
			sub_corr_info->setEdgeValue(edge, PVAD2GViewEdge());
		}

		sub_graph->delNode(node);
	}

	return sub_graph;
}
#endif

static tlp::node serializable_sub_graph_get_view(tlp::Graph* g, tlp::StringProperty* sub_corr_info, std::string const& view)
{
	tlp::node node;

	// Looks for 'view' in the 'g'
	forEach(node, g->getNodes()) {
		if(view == sub_corr_info->getNodeValue(node)) {
			returnForEach(node);
		}
	}

	return TLP_NODE_INVALID;
}

static tlp::node serializable_sub_graph_add_view(tlp::Graph* g, tlp::StringProperty* sub_corr_info, std::string const& view)
{
	tlp::node node = serializable_sub_graph_get_view(g, sub_corr_info, view);

	if (node.isValid()) {
		return node;
	}

	node = g->addNode();
	if (!node.isValid()) {
		return TLP_NODE_INVALID;
	}

	sub_corr_info->setNodeValue(node, view);

	return node;
}

tlp::Graph* Picviz::PVAD2GView::get_serializable_sub_graph() const
{
	tlp::Graph* sub_graph = tlp::newGraph();
	tlp::copyToGraph(sub_graph, _graph);

	PVAD2GViewCorrelationProperty* corr_info = sub_graph->getLocalProperty<PVAD2GViewCorrelationProperty>(TLP_CORR_PROPERTY);
	tlp::StringProperty* sub_corr_info = sub_graph->getLocalProperty<tlp::StringProperty>(TLP_STR_CORR_PROPERTY);

	// Convert every nodes
	QList<tlp::node> nodes_to_del;
	tlp::node node;
	forEach(node, sub_graph->getNodes()) {
		if (!node.isValid()) {
			nodes_to_del.push_back(node);
			continue;
		}
		Picviz::PVView const* view = corr_info->getNodeValue(node);
		if (!view || view->get_last_so().expired()) {
			nodes_to_del.push_back(node);
			continue;
		}

		sub_corr_info->setNodeValue(node, corr_info->getNodeStringValue(node));
	}

	foreach(node, nodes_to_del) {
		sub_graph->delNode(node);
	}

	// Visit each edges, and serialize them.
	tlp::edge edge;
	forEach(edge, sub_graph->getEdges()) {
		if (edge.isValid()) {
			sub_corr_info->setEdgeValue(edge, corr_info->getEdgeStringValue(edge));
		}
	}

	// Remove correlationProperty
	sub_graph->delLocalProperty(TLP_CORR_PROPERTY);

	return sub_graph;
}

/******************************************************************************
 *
 * Picviz::PVAD2GView::get_view
 *
 *****************************************************************************/
Picviz::PVView* Picviz::PVAD2GView::get_view(tlp::node node)
{
	if(_graph->isElement(node) == false)
		return 0;

	return _corr_info->getNodeValue(node);
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
 * Picviz::PVAD2GView::set_edge_f
 *
 *****************************************************************************/
tlp::edge Picviz::PVAD2GView::set_edge_f(const tlp::node na,
                                         const tlp::node nb,
                                         PVCombiningFunctionView_p cfview)
{
	tlp::edge edge = _graph->existEdge(na, nb, true);

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


	return _corr_info->getEdgeValue(edge);
}

/******************************************************************************
 *
 * Picviz::PVAD2GView::get_edge_views
 *
 *****************************************************************************/
Picviz::PVAD2GView::graph_edge_views_t Picviz::PVAD2GView::get_edge_views(const tlp::edge edge) const
{
	std::pair<tlp::node, tlp::node> res = _graph->ends(edge);

	return Picviz::PVAD2GView::graph_edge_views_t(_corr_info->getNodeValue(res.first),
	                                              _corr_info->getNodeValue(res.second));
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
		f(*_corr_info->getEdgeValue(edge),
		  *_corr_info->getNodeValue(a),
		  *_corr_info->getNodeValue(b));
	}
}

void Picviz::PVAD2GView::set_selected_edge(Picviz::PVView* view_src, Picviz::PVView* view_dst)
{
	tlp::node src = get_graph_node(view_src);
	tlp::node dst = get_graph_node(view_dst);
	tlp::edge edge = _graph->existEdge(src, dst);

	tlp::ColorProperty* color_property = _graph->getProperty<tlp::ColorProperty>("viewColor");
	color_property->setAllEdgeValue(tlp::Color(142, 142, 142));
	color_property->setEdgeValue(edge, tlp::Color(255, 0, 0));
}

/******************************************************************************
 *
 * Picviz::PVAD2GView::visit_from_view_f
 *
 *****************************************************************************/
void Picviz::PVAD2GView::visit_from_view_f(Picviz::PVView *view, graph_func_t const& f, QList<Picviz::PVView*>* changed_views) const
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
		va = _corr_info->getNodeValue(node);

		forEach(edge, _graph->getOutEdges(node)) {
			next = _graph->target(edge);

			// a PVView is only updated once
			if (visited.find(next) != visited.end())
				continue;

			vb = _corr_info->getNodeValue(next);
			PVLOG_INFO("propagating from view %p to view %p\n",
			            va, vb);
			cfview_p = _corr_info->getEdgeValue(edge);

			f(*cfview_p, *va, *vb);

			if (changed_views) {
				changed_views->push_back(vb);
			}

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
 * Picviz::PVAD2GView::get_edg_graph->delNode(node);es_count
 *
 *****************************************************************************/
size_t Picviz::PVAD2GView::get_edges_count() const
{
	return _graph->numberOfEdges();
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
		if(view == _corr_info->getNodeValue(node)) {
			returnForEach(node);
		}
	}

	return TLP_NODE_INVALID;
}

/******************************************************************************
 *
 * Picviz::PVAD2GView::del_edge
 *
 *****************************************************************************/
void Picviz::PVAD2GView::del_edge_f(Picviz::PVView* va, Picviz::PVView* vb)
{
	tlp::node na = get_graph_node(va);
	tlp::node nb = get_graph_node(vb);
	if (na != TLP_NODE_INVALID && nb != TLP_NODE_INVALID) {
		tlp::edge edge = _graph->existEdge(na, nb);
		if (edge != TLP_EDGE_INVALID) {
			_graph->delEdge(edge);
		}
	}
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
		// skipping paths with loop
		graph_path_t::iterator it = std::find(path.begin(), path.end(), next);
		if (it != path.end())
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

void Picviz::__impl::f_update_sel::operator()(Picviz::PVCombiningFunctionView& cf, Picviz::PVView& va, Picviz::PVView& vb) const
{
	PVSelection sel = cf(va, vb);
	vb.set_selection_view(sel);
}

void Picviz::PVAD2GView::load_from_file(QString const& path)
{
	PVCore::PVSerializeArchive_p ar(new PVCore::PVSerializeArchive(path, PVCore::PVSerializeArchive::read, PICVIZ_ARCHIVES_VERSION));
	ar->get_root()->object("ad2g", *this);
	ar->finish();
}

void Picviz::PVAD2GView::save_to_file(QString const& path)
{
	PVCore::PVSerializeArchive_p ar(new PVCore::PVSerializeArchive(path, PVCore::PVSerializeArchive::write, PICVIZ_ARCHIVES_VERSION));
	ar->get_root()->object("ad2g", *this);
	ar->finish();
}

void Picviz::PVAD2GView::serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	// Get the tulip graph file
	QString buf_file;
	so.buffer_path("graph", buf_file);

	// And import it !
	if (_graph) {
		delete _graph;
	}
	std::string buf_file_cstr(qPrintable(buf_file));
	_graph = tlp::loadGraph(buf_file_cstr);
	if (!_graph) {
		// Fail to load correlation graph !
		_graph = tlp::newGraph();
		_corr_info = _graph->getLocalProperty<PVAD2GViewCorrelationProperty>(TLP_CORR_PROPERTY);
		return;
	}
	// TLP graph properties
	_corr_info = _graph->getLocalProperty<PVAD2GViewCorrelationProperty>(TLP_CORR_PROPERTY);
	tlp::StringProperty* sub_corr_info = _graph->getLocalProperty<tlp::StringProperty>(TLP_STR_CORR_PROPERTY);

	// Convert edges to PVCombiningFunctionView
	tlp::edge e;
	forEach(e, _graph->getEdges()) {
		if (!e.isValid()) {
			continue;
		}
		_corr_info->setEdgeStringValue(e, sub_corr_info->getEdgeValue(e));
	}

	// Then, we need to get the real Picviz::PVView pointers.
	QList<tlp::node> nodes_to_del;
	tlp::node node;
	forEach(node, _graph->getNodes()) {
		std::string view_path = sub_corr_info->getNodeValue(node);
		if (view_path.size() == 0) {
			nodes_to_del.push_back(node);
			continue;
		}
		QString view_path_qs(view_path.c_str());
		bool view_valid = false;
		if (so.object_exists_by_path(view_path_qs)) {
			PVCore::PVSerializeObject_p view_so = so.get_object_by_path(view_path_qs);
			Picviz::PVView* view_obj = view_so->bound_obj_as<Picviz::PVView>();
			if (view_obj) {
				view_valid = true;
				_corr_info->setNodeValue(node, view_obj);
			}
		}
		if (!view_valid) {
			nodes_to_del.push_back(node);
		}
	}

	_graph->delLocalProperty(TLP_STR_CORR_PROPERTY);

	// Remove invalid nodes
	/*foreach(node, nodes_to_del) {
		del_view_by_node(node);
	}*/
}

void Picviz::PVAD2GView::serialize_write(PVCore::PVSerializeObject& so)
{
	assert(_graph);

	// Get the serializable sub graph (according to the view that have been serialized)
	tlp::Graph* sub_graph = get_serializable_sub_graph();

	std::stringstream ss_graph;
	{
		// Tulip graph export
		// Based on tlp::saveGraph function
		tlp::DataSet data;
		if (!tlp::exportGraph(sub_graph, ss_graph, "tlp", data, 0)) {
			delete sub_graph;
			return;
		}
	}
	delete sub_graph;

	std::string serialized_graph = ss_graph.str();

	so.buffer("graph", (void*) serialized_graph.c_str(), serialized_graph.size());
}
