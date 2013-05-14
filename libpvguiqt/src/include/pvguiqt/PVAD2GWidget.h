/**
 * \file PVAD2GWidget.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVAD2BWIDGET_H
#define PICVIZ_PVAD2GWIDGET_H

#include <QTableWidget>
#include <QWidget>
#include <QTableWidget>
#include <QTableWidget>

#include <pvkernel/core/general.h>

#include <picviz/PVAD2GView_types.h>
#include <picviz/PVRoot_types.h>
#include <picviz/widgets/PVAD2GEdgeEditor.h>

#include <tulip/NodeLinkDiagramComponent.h>
#include <tulip/Observable.h>
#include <tulip/TlpQtTools.h>

namespace PVWidgets {
class PVAD2GListEdgesWidget;
}

namespace PVGuiQt {

class PVAD2GWidget;

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
class AD2GNodeLinkDiagramComponent;

class LibPicvizDecl PVAD2GWidget : public QWidget, public tlp::Observable
{
	Q_OBJECT;

	typedef std::set<tlp::Observable*>::iterator ObserverIterator;

public:
	PVAD2GWidget(Picviz::PVAD2GView_p ad2g, Picviz::PVRoot& root, QWidget* parent = NULL);
	virtual ~PVAD2GWidget();

public:
	tlp::node add_view(QPoint pos, Picviz::PVView* view);
	tlp::edge add_combining_function(const tlp::node source, const tlp::node target);
	void edit_combining_function(tlp::edge edge, tlp::node src, tlp::node dst);
	//void select_edge(Picviz::PVView* view_src, Picviz::PVView* view_dst);

	__impl::PVTableWidget* get_table() { return _table; }
	Picviz::PVAD2GView& get_ad2g() { return *_ad2g; }

public slots:
	void update(ObserverIterator /*begin*/, ObserverIterator /*end*/);
	void observableDestroyed(tlp::Observable* /*observable*/) {}

	void add_view_Slot(QObject* mouse_event);
	void remove_view_Slot(int node);
	void remove_combining_function_Slot(int edge);

	void update_list_views();
	void update_list_edges();

	AD2GInteractor* getInteractor() { return _ad2g_interactor; }
	AD2GNodeLinkDiagramComponent* getNodeLinkView() { return _nodeLinkView; }

private:
	void init_table();
	void initObservers();
	void clearObservers();
	void set_enabled_view_item_in_table(Picviz::PVView* view, bool enabled);

	Picviz::PVRoot& get_root() const { return _root; }

private:
	Picviz::PVAD2GView_p _ad2g;
	AD2GNodeLinkDiagramComponent* _nodeLinkView;
	QWidget* _widget;
	QMainWindow* _mw;
	tlp::Graph* _graph;
	QToolBar* _toolBar;
	__impl::PVTableWidget* _table;
	PVWidgets::PVAD2GEdgeEditor* _edge_editor;

	AD2GInteractor* _ad2g_interactor;
	PVWidgets::PVAD2GListEdgesWidget* _list_edges_widget ;

	Picviz::PVRoot& _root;
};

namespace __impl {

class FilterDropEvent : public QObject
{
public:
	FilterDropEvent(PVGuiQt::PVAD2GWidget* widget) : QObject(widget), _widget(widget) {}
	bool eventFilter(QObject *object, QEvent *event);
private:
	PVGuiQt::PVAD2GWidget* _widget;
};

}

}

#endif //PICVIZ_PVAD2GWIDGET_H
