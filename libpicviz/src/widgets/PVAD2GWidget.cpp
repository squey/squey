#include <picviz/widgets/PVAD2GWidget.h>

Picviz::PVAD2GWidget::PVAD2GWidget(PVAD2GView& ad2g, QWidget* parent /*= NULL*/) :
	_ad2g(ad2g)
{
	_nodeLinkView = new tlp::NodeLinkDiagramComponent();

	tlp::DataSet dataSet;
	dataSet.set("arrow", true);

	PVLOG_INFO("glMainWidget=%x\n", _nodeLinkView->getGlMainWidget());
	openGraphOnGlMainWidget(_ad2g.get_graph(), &dataSet, _nodeLinkView->getGlMainWidget());
}

Picviz::PVAD2GWidget::~PVAD2GWidget()
{
	delete _nodeLinkView;
}

