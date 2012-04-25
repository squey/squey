#ifndef PICVIZ_PVAD2GLISTEDGESWIDGET_H
#define PICVIZ_PVAD2GLISTEDGESWIDGET_H

#include <pvkernel/core/general.h>

#include <picviz/PVSelRowFilteringFunction_types.h>

#include <QTableWidget>
#include <QStackedWidget>
#include <QWidget>
#include <QAction>

namespace Picviz {
class PVAD2GView;
class PVCombiningFunctionView;
class PVView;
}

namespace PVWidgets {

class PVAD2GEdgeEditor;
class PVAD2GFunctionPropertiesWidget;

class LibPicvizDecl PVAD2GListEdgesWidget: public QWidget
{
	Q_OBJECT

public:
	PVAD2GListEdgesWidget(Picviz::PVAD2GView& graph, QWidget* parent = NULL);
	void select_row(int src_view_id, int dst_view_id);

public slots:
	void update_list_edges();
	void update_fonction_properties(const Picviz::PVView& src_view, const Picviz::PVView& dst_view, Picviz::PVSelRowFilteringFunction_p& rff);
	void update_edge_editor_Slot(const Picviz::PVSelRowFilteringFunction_p & rff);
	void selection_changed_Slot(QTableWidgetItem* cur, QTableWidgetItem* prev);
	void remove_Slot();

protected slots:
	void show_edge(int row, int column = 0);

protected:
	Picviz::PVAD2GView& _graph;
	QTableWidget* _edges_table;
	PVAD2GFunctionPropertiesWidget* _function_properties_widget;
	PVAD2GEdgeEditor* _edge_properties_widget;
	Picviz::PVCombiningFunctionView* _cur_cf;
	QAction* _removeAct;
};

}

#endif
