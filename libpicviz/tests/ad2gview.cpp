
#include <picviz/PVView.h>
#include <picviz/PVAD2GView.h>
#include <picviz/PVCombiningFunctionView.h>
#include <picviz/PVCombiningFunctionView_types.h>

#include <tulip/TlpTools.h>

#include <iostream>

#define REPORT_RESULT(variable,value)	  \
	if ((variable) == (value)) \
		std::cout << "  WRONG" << std::endl; \
	else \
		std::cout << "  correct" << std::endl \

int main(void)
{
	tlp::initTulipLib();

	Picviz::PVView *va = new Picviz::PVView();
	Picviz::PVView *vb = new Picviz::PVView();
	Picviz::PVAD2GView ad2gv;
	Picviz::PVCombiningFunctionView_p cfva(new Picviz::PVCombiningFunctionView());
	Picviz::PVCombiningFunctionView_p cfvb(new Picviz::PVCombiningFunctionView());

	bool ret;

	std::cout << "first call to PVAD2GView::add_node(va)" << std::endl;
	ret = ad2gv.add_node(va);
	REPORT_RESULT(ret, false);


	std::cout << "second call to PVAD2GView::add_node(va)" << std::endl;
	ret = ad2gv.add_node(va);
	REPORT_RESULT(ret, true);

	std::cout << "first call to PVAD2GView::add_node(vb)" << std::endl;
	ret = ad2gv.add_node(vb);
	REPORT_RESULT(ret, false);

	std::cout << "first call to PVAD2GView::set_edge_f(va, vb, cfva)" << std::endl;
	ret = ad2gv.set_edge_f(va, vb, cfva);
	REPORT_RESULT(ret, false);

	std::cout << "second call to PVAD2GView::set_edge_f(va, vb, cfva)" << std::endl;
	ret = ad2gv.set_edge_f(va, vb, cfva);
	REPORT_RESULT(ret, false);

	std::cout << "first call to PVAD2GView::set_edge_f(va, vb, cfvb)" << std::endl;
	ret = ad2gv.set_edge_f(va, vb, cfvb);
	REPORT_RESULT(ret, false);

	return 0;
}
