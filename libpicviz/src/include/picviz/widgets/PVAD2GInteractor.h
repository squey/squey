#ifndef __PVAD2GINTERACTOR_H__
#define __PVAD2GINTERACTOR_H__

#include <QtGui>
#include <QtCore>

#include <tulip/Interactor.h>
#include <tulip/InteractorManager.h>
#include <tulip/InteractorComponent.h>
#include <tulip/InteractorChainOfResponsibility.h>
#include <tulip/Observable.h>
#include <tulip/Coord.h>
#include <tulip/InteractorComponent.h>
#include <tulip/ObservableProperty.h>
#include <tulip/ObservableGraph.h>
#include <tulip/TlpQtTools.h>
#include <tulip/MouseInteractors.h>
#include <tulip/MouseEdgeBendEditor.h>

#include <pvkernel/core/general.h>
#include <picviz/widgets/PVMouseSelectionEditor.h>
#include <picviz/widgets/PVMouseSelector.h>


namespace PVWidgets {

class PVAD2GWidget;

/**
 * \class AD2GInteractorComponent
 */
class AD2GInteractorComponent : public tlp::InteractorComponent, public tlp::PropertyObserver, public tlp::GraphObserver
{
	Q_OBJECT

public:
	AD2GInteractorComponent(PVAD2GWidget* widget, tlp::GlMainWidget* glMainWidget, Qt::MouseButton button = Qt::LeftButton, Qt::KeyboardModifier modifier = Qt::NoModifier);
	~AD2GInteractorComponent() {}
	bool eventFilter(QObject* widget, QEvent* e);
	tlp::InteractorComponent* clone() { return new AD2GInteractorComponent(_widget, _glMainWidget, mButton, kModifier); }
	void initObserver(tlp::Graph *newGraph);
	void clearObserver();
	bool draw(tlp::GlMainWidget*);

protected:
	virtual void addLink(QObject *, const tlp::node source, const tlp::node dest);

private:
	void abortEdgeTracing();
	void mMouseTranslate(tlp::node, double newX, double newY, GlMainWidget *glMainWidget);
	void update_selection(tlp::Graph* graph, tlp::GlMainWidget* glMainWidget, bool hoveringOverNode, bool hoveringOverEdge);

protected:
	Qt::MouseButton mButton;
	Qt::KeyboardModifier kModifier;
	Qt::KeyboardModifiers mousePressModifier;
	PVAD2GWidget* _widget;
	QSignalMapper* _addNodeSignalMapper;
	QSignalMapper* _deleteNodeSignalMapper;
	QSignalMapper* _deleteEdgeSignalMapper;
	tlp::GlMainWidget* _glMainWidget;

	std::vector<tlp::Coord> _bends;
	tlp::LayoutProperty* _layoutProperty;
	tlp::Coord _startPos, _curPos;
	bool _edge_started;
	tlp::node _source;

	tlp::node _tmpNode;
	tlp::edge _tmpEdge;
	tlp::ElementType _type;
	Coord _editPosition;
	bool _translation_started;
};

/**
 * \class AD2GInteractor
 */
class AD2GInteractor : public tlp::InteractorChainOfResponsibility
{
	Q_OBJECT

public:
	AD2GInteractor(PVAD2GWidget* widget, tlp::GlMainWidget* glMainWidget):
		tlp::InteractorChainOfResponsibility("", "Edit graph"),
		_widget(widget),
		_glMainWidget(glMainWidget)
	{
		construct();
	}
	void construct()
	{
		_component1 = new AD2GInteractorComponent(_widget, _glMainWidget);
		pushInteractorComponent(_component1);
	}
	virtual bool isCompatible(const std::string &/*viewName*/) {return true;}

private:
	AD2GInteractorComponent* _component1;
	PVAD2GWidget* _widget;
	tlp::GlMainWidget* _glMainWidget;
};

}

#endif // __PVAD2GINTERACTOR_H__
