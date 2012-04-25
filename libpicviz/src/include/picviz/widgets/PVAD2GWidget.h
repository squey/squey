#ifndef PICVIZ_PVAD2BWIDGET_H
#define PICVIZ_PVAD2GWIDGET_H

#include <QtGui>

#include <tulip/NodeLinkDiagramComponent.h>
#include <tulip/Observable.h>
#include <tulip/TlpQtTools.h>

#include <pvkernel/core/general.h>
#include <picviz/PVAD2GView.h>

#include <picviz/widgets/PVAD2GEdgeEditor.h>


namespace PVWidgets {

class PVAD2GWidget;
class PVAD2GListEdgesWidget;

namespace __impl {

class PVTableWidget : public QTableWidget
{
public:
	PVTableWidget(QWidget* parent = 0) : QTableWidget(parent) { setEditTriggers(QTableWidget::NoEditTriggers); }
	void mousePressEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
private:
	QPoint _dragStartPosition;
};

}

class AD2GInteractor;
class AD2GInteractor2;

class AD2GNodeLinkDiagramComponent : public tlp::NodeLinkDiagramComponent
{
public:
	// Disable view standard context menu
	virtual void buildContextMenu(QObject*, QContextMenuEvent*, QMenu*){};
	virtual void computeContextMenuAction(QAction*){};
};

class LibPicvizDecl PVAD2GWidget : public QWidget, public tlp::Observable
{
	Q_OBJECT;

	typedef std::set<tlp::Observable*>::iterator ObserverIterator;

public:
	PVAD2GWidget(Picviz::PVAD2GView& ad2g, QWidget* parent = NULL);
	~PVAD2GWidget();

public:
	tlp::node add_view(QPoint pos, Picviz::PVView* view);
	tlp::edge add_combining_function(const tlp::node source, const tlp::node target);
	void edit_combining_function(tlp::edge edge, tlp::node src, tlp::node dst);
	//void select_edge(Picviz::PVView* view_src, Picviz::PVView* view_dst);
	void set_edit_graph(bool edit_graph);
	void highlightViewItem(tlp::node n);

	__impl::PVTableWidget* get_table() { return _table; }
	Picviz::PVAD2GView& get_ad2g() { return _ad2g; }

public slots:
	void update_interactor_Slot();
	void update(ObserverIterator /*begin*/, ObserverIterator /*end*/);
	void observableDestroyed(tlp::Observable* /*observable*/) {}

	void add_view_Slot(QObject* mouse_event);
	void remove_view_Slot(int node);
	void remove_combining_function_Slot(int edge);

	void update_list_views();

	AD2GInteractor* getInteractor() { return _ad2g_interactor; }
	AD2GInteractor2* getInteractor2() { return _ad2g_interactor2; }
	AD2GNodeLinkDiagramComponent* getNodeLinkView() { return _nodeLinkView; }

private:
	void init_table();
	void initObservers();
	void clearObservers();

private:
	Picviz::PVAD2GView& _ad2g;
	AD2GNodeLinkDiagramComponent* _nodeLinkView;
	QWidget* _widget;
	QMainWindow* _mw;
	tlp::Graph* _graph;
	QToolBar* _toolBar;
	__impl::PVTableWidget* _table;
	PVWidgets::PVAD2GEdgeEditor* _edge_editor;

	QRadioButton* _radio_edit_graph;
	QRadioButton* _radio_edit_layout;

	AD2GInteractor* _ad2g_interactor;
	AD2GInteractor2* _ad2g_interactor2;
	PVAD2GListEdgesWidget* _list_edges_widget ;
};

namespace __impl {

class FilterDropEvent : public QObject
{
public:
	FilterDropEvent(PVWidgets::PVAD2GWidget* widget) : QObject(widget), _widget(widget) {}
	bool eventFilter(QObject *object, QEvent *event);
private:
	PVWidgets::PVAD2GWidget* _widget;
};

}

}

#endif //PICVIZ_PVAD2GWIDGET_H
