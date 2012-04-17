
#include <picviz/PVView.h>
#include <picviz/PVAD2GView.h>
#include <picviz/PVCombiningFunctionView.h>
#include <picviz/PVCombiningFunctionView_types.h>
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
	tlp::node N_INVAL;
	tlp::node na, nb, nc, nd;
	tlp::edge E_INVAL;
	tlp::edge e1, e2, e3, e4;

	/* some test for ::add_view()
	 */
	std::cout << "-------------------------------------------------------------------------------" << std::endl;
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

	std::cout << "-------------------------------------------------------------------------------" << std::endl;
	/* we create an usable PVAD2GView
	 */
	ad2gv = new Picviz::PVAD2GView(scene.get());

	// va, vb, vc, vd  are already allocated
	na = ad2gv->add_view(va);
	nb = ad2gv->add_view(vb);
	nc = ad2gv->add_view(vc);
	nd = ad2gv->add_view(vd);

	// set views name
	std::cout << "va = " << va << std::endl;
	std::cout << "vb = " << vb << std::endl;
	std::cout << "vc = " << vc << std::endl;
	std::cout << "vd = " << vd << std::endl;

	std::cout << "::run() with no correlation: no output" << std::endl;

	std::cout << "-------------------------------------------------------------------------------" << std::endl;
	ad2gv->run(va);

	// add some CFV
	e1 = ad2gv->set_edge_f(va, vb, cfv1);
	e2 = ad2gv->set_edge_f(va, vc, cfv2);
	e3 = ad2gv->set_edge_f(vb, vd, cfv3);
	e4 = ad2gv->set_edge_f(vd, vb, cfv4);

	std::cout << "-------------------------------------------------------------------------------" << std::endl;
	std::cout << "::run(va) with correlations: output needed" << std::endl;
	ad2gv->run(va);

	std::cout << "-------------------------------------------------------------------------------" << std::endl;
	std::cout << "::run(vb) with correlations: output needed" << std::endl;
	ad2gv->run(vb);

	std::cout << "-------------------------------------------------------------------------------" << std::endl;
	std::cout << "::run(vd) with correlations: output needed" << std::endl;
	ad2gv->run(vd);

	return 0;
}
