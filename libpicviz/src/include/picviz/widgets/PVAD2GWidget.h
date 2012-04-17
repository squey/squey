#ifndef PICVIZ_PVAD2BWIDGET_H
#define PICVIZ_PVAD2GWIDGET_H

#include <QtGui>

#include <tulip/NodeLinkDiagramComponent.h>
#include <tulip/Observable.h>
#include <tulip/TlpQtTools.h>

#include <pvkernel/core/general.h>
#include <picviz/PVAD2GView.h>

namespace Picviz {

namespace __impl {

class PVAD2GWidget;

class PVTableWidget : public QTableWidget
{
public:
	PVTableWidget(QWidget* parent = 0) : QTableWidget(parent) {}
	void mousePressEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
private:
	QPoint _dragStartPosition;
};

}


class AD2GNodeLinkDiagramComponent : public tlp::NodeLinkDiagramComponent
{
public:
	// Disable view standard context menu
	virtual void buildContextMenu(QObject*, QContextMenuEvent*, QMenu*){};
	virtual void computeContextMenuAction(QAction*){};
};

class /*LibPicvizExport*/ PVAD2GWidget : public QWidget, public tlp::Observable
{
	Q_OBJECT;

	typedef std::set<tlp::Observable*>::iterator ObserverIterator;

public:
	PVAD2GWidget(PVAD2GView& ad2g, QMainWindow* mw = NULL);
	~PVAD2GWidget();

public:
	void add_view(QPoint pos, PVView* view);
	__impl::PVTableWidget* get_table() { return _table; }
	PVAD2GView& get_ad2g() { return _ad2g; }

public slots:
	void change_interactor_slot();
	void update(ObserverIterator /*begin*/, ObserverIterator /*end*/);
	void observableDestroyed(tlp::Observable* /*observable*/) {}

	void add_view_Slot(QObject* mouse_event);
	void remove_view_Slot(int node);
	void remove_combining_function_Slot(int edge);

private:
	void fill_table();
	void init_toolbar();
	void initObservers();
	void clearObservers();

private:
	PVAD2GView& _ad2g;
	AD2GNodeLinkDiagramComponent* _nodeLinkView;
	QWidget* _widget;
	QMainWindow* _mw;
	tlp::Graph* _graph;
	QToolBar* _toolBar;
	__impl::PVTableWidget* _table;
};

namespace __impl {

class FilterDropEvent : public QObject
{
public:
	FilterDropEvent(Picviz::PVAD2GWidget* widget) : QObject(widget), _widget(widget) {}
	bool eventFilter(QObject *object, QEvent *event);
private:
	Picviz::PVAD2GWidget* _widget;
};

}

}

#endif //PICVIZ_PVAD2GWIDGET_H
