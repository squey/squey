#ifndef PICVIZ_PVAD2GLISTEDGESWIDGET_H
#define PICVIZ_PVAD2GLISTEDGESWIDGET_H

#include <pvkernel/core/general.h>
#include <picviz/widgets/PVAD2GFunctionPropertiesWidget.h>

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
	void update_fonction_properties(const Picviz::PVView& src_view, const Picviz::PVView& dst_view, Picviz::PVSelRowFilteringFunction_p& rff);

protected:
	Picviz::PVAD2GView& _graph;
	QTableWidget* _edges_table;
	PVAD2GFunctionPropertiesWidget* _function_properties_widget;
};

}

#endif
