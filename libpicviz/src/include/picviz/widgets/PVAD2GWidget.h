#ifndef PICVIZ_PVAD2BWIDGET_H
#define PICVIZ_PVAD2GWIDGET_H

#include <QtGui>

#include <tulip/NodeLinkDiagramComponent.h>
#include <tulip/TlpQtTools.h>

#include <pvkernel/core/general.h>
#include <picviz/PVAD2GView.h>

namespace Picviz {

class /*LibPicvizExport*/ PVAD2GWidget : public QWidget
{
	Q_OBJECT;
public:
	PVAD2GWidget(PVAD2GView& ad2g, QWidget* parent = NULL);
	~PVAD2GWidget();

private:
	PVAD2GView& _ad2g;
	tlp::NodeLinkDiagramComponent* _nodeLinkView;
};

}

#endif //PICVIZ_PVAD2GWIDGET_H
