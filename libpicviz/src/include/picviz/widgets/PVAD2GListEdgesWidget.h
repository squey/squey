#ifndef PICVIZ_PVAD2GLISTEDGESWIDGET_H
#define PICVIZ_PVAD2GLISTEDGESWIDGET_H

#include <pvkernel/core/general.h>

#include <QTableWidget>
#include <QWidget>

namespace Picviz {
class PVAD2GView;
}

namespace PVWidgets {

class LibPicvizDecl PVAD2GListEdgesWidget: public QWidget
{
	Q_OBJECT

public:
	PVAD2GListEdgesWidget(Picviz::PVAD2GView& graph, QWidget* parent = NULL);

public slots:
	void update_list_edges();

protected:
	Picviz::PVAD2GView& _graph;
	QTableWidget* _edges_table;
};

}

#endif
