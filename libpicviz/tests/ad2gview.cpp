#include "test-env.h"
#include <picviz/PVView.h>
#include <picviz/PVAD2GView.h>
#include <picviz/PVCombiningFunctionView.h>
#include <picviz/PVCombiningFunctionView_types.h>
#include <picviz/PVSelRowFilteringFunction.h>
#include <picviz/PVTFViewRowFiltering.h>
#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>

#include <tulip/Graph.h>

#include <iostream>

#define REPORT_RESULT(test)	  \
	if ((test)) \
		std::cout << "  correct: (" #test ")" << std::endl; \
	else \
		std::cout << "  WRONG: not (" #test ")" << std::endl \

int main(void)
{
	init_env();
	Picviz::PVView *va = new Picviz::PVView();
	Picviz::PVView *vb = new Picviz::PVView();
	Picviz::PVView *vc = new Picviz::PVView();
	Picviz::PVView *vd = new Picviz::PVView();

	Picviz::PVRoot_p root(new Picviz::PVRoot());
	Picviz::PVScene_p scene(new Picviz::PVScene("scene", root.get()));


	Picviz::PVAD2GView *ad2gv = new Picviz::PVAD2GView(scene.get());
	Picviz::PVCombiningFunctionView_p cfv1(new Picviz::PVCombiningFunctionView());
	Picviz::PVCombiningFunctionView_p cfv2(new Picviz::PVCombiningFunctionView());
	Picviz::PVCombiningFunctionView_p cfv3(new Picviz::PVCombiningFunctionView());
	Picviz::PVCombiningFunctionView_p cfv4(new Picviz::PVCombiningFunctionView());
	tlp::node NODE_INVAL;
	tlp::node na, nb, nc, nd;
	tlp::edge EDGE_INVAL;
	tlp::edge e1, e2, e3, e4;
	bool check_ret;

	/* some test for ::add_view()
	 */
	std::cout << "-------------------------------------------------------------------------------" << std::endl;
	std::cout << "first call to PVAD2GView::add_view(va)" << std::endl;
	na = ad2gv->add_view(va);
	REPORT_RESULT(na != NODE_INVAL);

	std::cout << "second call to PVAD2GView::add_view(va)" << std::endl;
	nb = ad2gv->add_view(va);
	REPORT_RESULT(nb == na);

	std::cout << "first call to PVAD2GView::add_view(vb)" << std::endl;
	nb = ad2gv->add_view(vb);
	REPORT_RESULT(nb != NODE_INVAL);

	/* some test for ::set_edge_f()
	 */
	std::cout << "first call to PVAD2GView::set_edge_f(va, vb, cfv1)" << std::endl;
	e1 = ad2gv->set_edge_f(va, vb, cfv1);
	REPORT_RESULT(e1 != EDGE_INVAL);

	std::cout << "second call to PVAD2GView::set_edge_f(va, vb, cfv1)" << std::endl;
	e2 = ad2gv->set_edge_f(va, vb, cfv1);
	REPORT_RESULT(e2 == e1);

	std::cout << "first call to PVAD2GView::set_edge_f(va, vb, cfv2)" << std::endl;
	e2 = ad2gv->set_edge_f(va, vb, cfv2);
	REPORT_RESULT(e2 == e1);

	Picviz::PVSelRowFilteringFunction_p rff = LIB_CLASS(Picviz::PVSelRowFilteringFunction)::get().get_class_by_name("axes_bind");
	assert(rff);
	cfv2->get_first_tf()->push_rff(rff);

	std::string cf_str;
	cfv2->to_string(cf_str);
	std::cout << "CF to string: " << std::endl << cf_str << std::endl;

	cfv1->from_string(cf_str);
	cf_str.clear();
	cfv1->to_string(cf_str);
	std::cout << "CF from->to string: " << std::endl << cf_str << std::endl;

	//ad2gv->save_to_file("/tmp/test.ad2g");

	delete ad2gv;

	std::cout << "-------------------------------------------------------------------------------" << std::endl;
	/* we create an usable PVAD2GView
	 */
	ad2gv = new Picviz::PVAD2GView(scene.get());

	// va, vb, vc, vd  are already allocated
	na = ad2gv->add_view(va);
	nb = ad2gv->add_view(vb);
	nc = ad2gv->add_view(vc);
	nd = ad2gv->add_view(vd);

	std::cout << "::run() with no correlation; there must be no output" << std::endl;
	ad2gv->run(va);

	// add some CFV
	e1 = ad2gv->set_edge_f(va, vb, cfv1);
	e2 = ad2gv->set_edge_f(va, vc, cfv2);
	e3 = ad2gv->set_edge_f(vb, vd, cfv3);
	e4 = ad2gv->set_edge_f(vd, vb, cfv4);

	std::cout << "-------------------------------------------------------------------------------" << std::endl;
	std::cout << "check_properties()" << std::endl;
	check_ret = ad2gv->check_properties();
	REPORT_RESULT(check_ret == true);

	std::cout << "adding an edge from vc to vd to make an invalid graph" << std::endl;
	e4 = ad2gv->set_edge_f(vc, vd, cfv4);
	std::cout << "check_properties()" << std::endl;
	check_ret = ad2gv->check_properties();
	REPORT_RESULT(check_ret == false);


	return 0;
}
