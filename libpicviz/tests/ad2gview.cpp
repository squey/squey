
#include <picviz/PVView.h>
#include <picviz/PVAD2GView.h>
#include <picviz/PVCombiningFunctionView.h>
#include <picviz/PVCombiningFunctionView_types.h>

#include <tulip/Graph.h>

#include <iostream>

#define REPORT_RESULT(test)	  \
	if ((test)) \
		std::cout << "  correct: (" #test ")" << std::endl; \
	else \
		std::cout << "  WRONG: not (" #test ")" << std::endl \

Picviz::PVAD2GView *create_graph()
{
	Picviz::PVAD2GView *ad2gv = new Picviz::PVAD2GView();
	/*
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
	*/
	return ad2gv;
}

int main(void)
{
	Picviz::PVView *va = new Picviz::PVView();
	Picviz::PVView *vb = new Picviz::PVView();
	Picviz::PVView *vc = new Picviz::PVView();
	Picviz::PVView *vd = new Picviz::PVView();

	Picviz::PVAD2GView *ad2gv = new Picviz::PVAD2GView();
	Picviz::PVCombiningFunctionView_p cfv1(new Picviz::PVCombiningFunctionView());
	Picviz::PVCombiningFunctionView_p cfv2(new Picviz::PVCombiningFunctionView());
	Picviz::PVCombiningFunctionView_p cfv3(new Picviz::PVCombiningFunctionView());
	Picviz::PVCombiningFunctionView_p cfv4(new Picviz::PVCombiningFunctionView());
	tlp::node N_INVAL;
	tlp::node na, nb, nc, nd;
	tlp::edge E_INVAL;
	tlp::edge e1, e2, e3, e4;

	/* some test for ::add_view()
	 */
	std::cout << "first call to PVAD2GView::add_view(va)" << std::endl;
	na = ad2gv->add_view(va);
	REPORT_RESULT(na != N_INVAL);

	std::cout << "second call to PVAD2GView::add_view(va)" << std::endl;
	nb = ad2gv->add_view(va);
	REPORT_RESULT(nb == na);

	std::cout << "first call to PVAD2GView::add_view(vb)" << std::endl;
	nb = ad2gv->add_view(vb);
	REPORT_RESULT(nb != N_INVAL);

	/* some test for ::set_edge_f()
	 */
	std::cout << "first call to PVAD2GView::set_edge_f(va, vb, cfv1)" << std::endl;
	e1 = ad2gv->set_edge_f(va, vb, cfv1);
	REPORT_RESULT(e1 != E_INVAL);

	std::cout << "second call to PVAD2GView::set_edge_f(va, vb, cfv1)" << std::endl;
	e2 = ad2gv->set_edge_f(va, vb, cfv1);
	REPORT_RESULT(e2 == e1);

	std::cout << "first call to PVAD2GView::set_edge_f(va, vb, cfv2)" << std::endl;
	e2 = ad2gv->set_edge_f(va, vb, cfv2);
	REPORT_RESULT(e2 == e1);

	delete ad2gv;

	/* we create an usable PVAD2GView
	 */
	ad2gv = new Picviz::PVAD2GView();

	// va, vb, vc, vd  are already allocated
	na = ad2gv->add_view(va);
	nb = ad2gv->add_view(vb);
	nc = ad2gv->add_view(vc);
	nd = ad2gv->add_view(vd);

	// set views name
	va->name = "Va";
	vb->name = "Vb";
	vc->name = "Vc";
	vd->name = "Vd";

	std::cout << "::run() with no correlation: no output" << std::endl;
	ad2gv->run(va);

	// add some CFV
	e1 = ad2gv->set_edge_f(va, vb, cfv1);
	e2 = ad2gv->set_edge_f(va, vc, cfv2);
	e3 = ad2gv->set_edge_f(vb, vd, cfv3);
	e4 = ad2gv->set_edge_f(vd, vb, cfv4);

	std::cout << "::run() with correlations: output needed" << std::endl;
	ad2gv->run(va);

	return 0;
}
