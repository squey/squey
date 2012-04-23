#ifndef PICVIZ_PVAD2GVIEW_H
#define PICVIZ_PVAD2GVIEW_H

#include <pvkernel/core/general.h>
#include <picviz/PVCombiningFunctionView.h>
#include <picviz/PVCombiningFunctionView_types.h>
#include <picviz/PVSelection.h>

#include <list>
#include <set>
#include <utility>

// forward declaration of tlp::Graph and tlp::node
namespace tlp {
class Graph;
struct node;
struct edge;
}

#include <picviz/PVScene.h>

namespace Picviz {

// forward declaration of the class for multi-source correlation property
class PVAD2GViewCorrelationProperty;
class PVView;

namespace __impl {
	struct f_update_sel
	{
		inline void operator()(Picviz::PVCombiningFunctionView& cf,
		                       Picviz::PVView& va, Picviz::PVView& vb) const
		{
			PVSelection sel = cf(va, vb);
			vb.set_selection_view(sel);
		}
	};

	struct f_pre_process
	{
		inline void operator()(Picviz::PVCombiningFunctionView& cf,
		                       Picviz::PVView& va, Picviz::PVView& vb) const
		{
			cf.pre_process(va, vb);
		}
	};
}

/**
 * \class PVAD2GView
 */
class PVAD2GView
{
	typedef boost::function<void(Picviz::PVCombiningFunctionView&, Picviz::PVView& va, Picviz::PVView& vb)> graph_func_t;
	typedef std::list<tlp::node> graph_path_t;
	typedef std::set<tlp::node> graph_visited_t;
	typedef std::pair<PVView*, PVView*> graph_edge_views_t;

public:
	PVAD2GView(Picviz::PVScene* scene);
	~PVAD2GView();

public:
	tlp::Graph *get_graph() { return _graph; }
	PVScene* get_scene() { return _scene; }

public:
	tlp::node add_view(Picviz::PVView *view);
	void del_view_by_node(tlp::node node);
	Picviz::PVView* get_view(tlp::node node);

public:
	tlp::edge set_edge_f(const Picviz::PVView *va, const Picviz::PVView *vb,
	                     PVCombiningFunctionView_p cfview);
	tlp::edge set_edge_f(const tlp::node na, const tlp::node nb,
	                     PVCombiningFunctionView_p cfview);
	Picviz::PVCombiningFunctionView_p get_edge_f(const tlp::edge edge) const;
	graph_edge_views_t get_edge_views(const tlp::edge edge) const;

public:
	void pre_process() const {
		visit_edges(__impl::f_pre_process());
	}

	void run(Picviz::PVView *view) const {
		visit_from_view(view, __impl::f_update_sel());
	}

	bool check_properties();

	template <class F>
	inline void visit_edges(F const& f) const
	{
		visit_edges_f(boost::bind<void>(f, _1, _2, _3));
	}

	size_t get_edges_count() const;


private:
	tlp::node get_graph_node(const Picviz::PVView *view) const;

	template <class F>
	inline void visit_from_view(Picviz::PVView *view, F const& f) const
	{
		visit_from_view_f(view, boost::bind<void>(f, _1, _2, _3));
	}

	void visit_edges_f(graph_func_t const& f) const;

	void visit_from_view_f(Picviz::PVView *view, graph_func_t const& f) const;

	int count_path_number(const tlp::node& a, const tlp::node& b) const;
	void count_path_number_rec(const tlp::node& a, const tlp::node& b, int& count, graph_path_t& path, graph_visited_t& visited)const;

private:
	/* graph tulip object */
	tlp::Graph *_graph;
	PVScene* _scene;
	/* graph's property */
	PVAD2GViewCorrelationProperty *_corr_info;
};

}

#endif // PICVIZ_PVAD2GVIEW_H
