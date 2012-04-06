#ifndef PICVIZ_PVAD2BWIDGET_H
#define PICVIZ_PVAD2GWIDGET_H

#include <QtGui>

#include <tulip/InteractorComponent.h>
#include "tulip/InteractorChainOfResponsibility.h"
#include <tulip/NodeLinkDiagramComponent.h>
#include <tulip/Observable.h>
#include <tulip/TlpQtTools.h>

#include <pvkernel/core/general.h>
#include <picviz/PVAD2GView.h>

namespace Picviz {

class AD2GNodeLinkDiagramComponent : public tlp::NodeLinkDiagramComponent
{
public:
	// Disable view standard context menu
	virtual void buildContextMenu(QObject* object, QContextMenuEvent* event, QMenu* contextMenu){};
	virtual void computeContextMenuAction(QAction *action){};
};

class /*LibPicvizExport*/ PVAD2GWidget : public QWidget, public tlp::Observable
{
	Q_OBJECT;

	typedef std::set<tlp::Observable*>::iterator ObserverIterator;

public:
	PVAD2GWidget(PVAD2GView& ad2g, QMainWindow* mw = NULL);
	~PVAD2GWidget();

	QWidget* get_widget();

public slots:
	void change_interactor_slot();
	void update(ObserverIterator begin, ObserverIterator end);
	void observableDestroyed(tlp::Observable* observable) {}

private:
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
};

/**
 * \class AD2GInteractorComponent
 */
class AD2GInteractorComponent : public tlp::InteractorComponent
{
	Q_OBJECT

public:
	AD2GInteractorComponent(PVAD2GWidget* widget, Qt::MouseButton button = Qt::LeftButton, Qt::KeyboardModifier modifier = Qt::NoModifier):_widget(widget){}
	~AD2GInteractorComponent() {}
	bool eventFilter(QObject* widget, QEvent* e);
	tlp::InteractorComponent* clone() { return new AD2GInteractorComponent(_widget, mButton, kModifier); }

protected:
	Qt::MouseButton mButton;
	Qt::KeyboardModifier kModifier;
	Qt::KeyboardModifiers mousePressModifier;
	PVAD2GWidget* _widget;
};

/**
 * \class AD2GInteractor
 */
class AD2GInteractor : public tlp::InteractorChainOfResponsibility
{
	Q_OBJECT

public:
	AD2GInteractor(PVAD2GWidget* widget):
		tlp::InteractorChainOfResponsibility("", "AD2GInteractor"),
		_widget(widget)
	{
		construct();
	}
	void construct()
	{
		_component1 = new AD2GInteractorComponent(_widget);
		pushInteractorComponent(_component1);
	}
	virtual bool isCompatible(const std::string &viewName) {return false;}

private:
	AD2GInteractorComponent* _component1;
	PVAD2GWidget* _widget;
};

}

#endif //PICVIZ_PVAD2GWIDGET_H
