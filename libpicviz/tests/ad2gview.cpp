
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

int main(void)
{
	Picviz::PVView *va = new Picviz::PVView();
	Picviz::PVView *vb = new Picviz::PVView();
	Picviz::PVAD2GView ad2gv;
	Picviz::PVCombiningFunctionView_p cfva(new Picviz::PVCombiningFunctionView());
	Picviz::PVCombiningFunctionView_p cfvb(new Picviz::PVCombiningFunctionView());
	tlp::node N_INVAL;
	tlp::node na, nb;
	tlp::edge E_INVAL;
	tlp::edge e1, e2;

	std::cout << "first call to PVAD2GView::add_view(va)" << std::endl;
	na = ad2gv.add_view(va);
	REPORT_RESULT(na != N_INVAL);

	std::cout << "second call to PVAD2GView::add_view(va)" << std::endl;
	nb = ad2gv.add_view(va);
	REPORT_RESULT(nb == na);

	std::cout << "first call to PVAD2GView::add_view(vb)" << std::endl;
	nb = ad2gv.add_view(vb);
	REPORT_RESULT(nb != N_INVAL);

	std::cout << "first call to PVAD2GView::set_edge_f(va, vb, cfva)" << std::endl;
	e1 = ad2gv.set_edge_f(va, vb, cfva);
	REPORT_RESULT(e1 != E_INVAL);

	std::cout << "second call to PVAD2GView::set_edge_f(va, vb, cfva)" << std::endl;
	e2 = ad2gv.set_edge_f(va, vb, cfva);
	REPORT_RESULT(e2 == e1);

	std::cout << "first call to PVAD2GView::set_edge_f(va, vb, cfvb)" << std::endl;
	e2 = ad2gv.set_edge_f(va, vb, cfvb);
	REPORT_RESULT(e2 == e1);

	return 0;
}
