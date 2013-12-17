/**
 * \file PVAD2GInteractor.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <tulip/NodeLinkDiagramComponent.h>
#include <tulip/GlLine.h>

#include <picviz/PVAD2GView.h>

#include <pvguiqt/PVAD2GWidget.h>
#include <pvguiqt/PVAD2GInteractor.h>

#define NODE_WIDTH 25
#define NODE_HEIGHT 25

PVGuiQt::AD2GInteractorComponent::AD2GInteractorComponent(PVGuiQt::PVAD2GWidget* widget, tlp::GlMainWidget* glMainWidget, Qt::MouseButton /*button*/ /*= Qt::LeftButton*/, Qt::KeyboardModifier /*modifier*/ /*= Qt::NoModifier*/) :
	_widget(widget),
	_glMainWidget(glMainWidget),
	_translation_started(false)
{
	_deleteNodeSignalMapper = new QSignalMapper(this);
	connect(_deleteNodeSignalMapper, SIGNAL(mapped(int)), _widget, SLOT(remove_view_Slot(int)));

	_addNodeSignalMapper = new QSignalMapper(this);
	connect(_addNodeSignalMapper, SIGNAL(mapped(QObject*)), _widget, SLOT(add_view_Slot(QObject*)));

	_deleteEdgeSignalMapper = new QSignalMapper(this);
	connect(_deleteEdgeSignalMapper, SIGNAL(mapped(int)), _widget, SLOT(remove_combining_function_Slot(int)));

	_edge_started = false;
}

bool PVGuiQt::AD2GInteractorComponent::eventFilter(QObject* widget, QEvent* e)
{
	if (!(e->type() == QEvent::MouseButtonPress || e->type() == QEvent::MouseButtonRelease || e->type() == QEvent::MouseMove || e->type() == QEvent::KeyPress || e->type() == QEvent::KeyRelease || e->type() == QEvent::MouseButtonDblClick)) {
		return false;
	}

	tlp::GlMainWidget* glMainWidget = (tlp::GlMainWidget* ) widget;
	tlp::Graph* graph = glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getGraph();
	tlp::LayoutProperty* mLayout = graph->getProperty<tlp::LayoutProperty>(glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getElementLayoutPropName());

	if (e->type() == QEvent::MouseButtonRelease)
	{
		_translation_started = false;
		//glMainWidget->setCursor(QCursor(Qt::ArrowCursor));
	}

	if (e->type() == QEvent::KeyPress) {
		QKeyEvent* qKeyEvent = (QKeyEvent*) e;
		if (qKeyEvent->key() == Qt::Key_Delete) {
			if (_type == tlp::NODE && _tmpNode != tlp::node()) {
				_widget->remove_view_Slot(_tmpNode);
				_tmpNode = tlp::node();
				return true;
			}
			else if (_type == tlp::EDGE && _tmpEdge != tlp::edge()) {
				_widget->remove_combining_function_Slot(_tmpEdge);
				_tmpEdge = tlp::edge();
				return true;
			}
		}
		else if (qKeyEvent->key() == Qt::Key_Escape){
			abort_edge_tracing();
			return true;
		}
		else if (qKeyEvent->key() == Qt::Key_Control) {
			glMainWidget->setCursor(QCursor(Qt::PointingHandCursor));
			return true;
		}
	}
	else if (e->type() == QEvent::KeyRelease)
	{
		glMainWidget->setCursor(QCursor(Qt::ArrowCursor));
		return true;
	}
	else {
		QMouseEvent* qMouseEv = (QMouseEvent*) e;

		bool hoveringOverNode = glMainWidget->doSelect(qMouseEv->x(), qMouseEv->y(), _type, _tmpNode, _tmpEdge) && _type == tlp::NODE;
		bool hoveringOverEdge = glMainWidget->doSelect(qMouseEv->x(), qMouseEv->y(), _type, _tmpNode, _tmpEdge) && _type == tlp::EDGE;

		if (qMouseEv->button()==Qt::LeftButton) {
			// Edit graph layout
			if (e->type() == QEvent::MouseButtonPress && qMouseEv->modifiers() & Qt::ControlModifier) {
				//glMainWidget->setCursor(QCursor(Qt::PointingHandCursor));
				_translation_started = true;

				_editPosition[0] = qMouseEv->x();
				_editPosition[1] = qMouseEv->y();
			}
			else if (qMouseEv->type() == QEvent::MouseButtonDblClick) {
				if (hoveringOverEdge) {
					tlp::node src = graph->source(_tmpEdge);
					tlp::node dst = graph->target(_tmpEdge);

					_widget->edit_combining_function(_tmpEdge, src, dst);
					return true;
				}
				return false;
			}
			// Start edge tracing
			else if(e->type() == QEvent::MouseButtonPress && !_edge_started) {
				if (hoveringOverNode) {
					_edge_started=true;
					initObserver(graph);
					_source=_tmpNode;
					_curPos=_startPos=mLayout->getNodeValue(_source);
					return true;
				}
				return false;
			}
			// Finish edge tracing
			else if(e->type() == QEvent::MouseButtonRelease && _edge_started) {
				if (hoveringOverNode) {
					tlp::Observable::holdObservers();
					_edge_started = false;
					clearObserver();

					if (_source == _tmpNode) {
						if(! _bends.empty()) {
							//QMessageBox* box = new QMessageBox(QMessageBox::Critical, tr("Invalid edge."), tr("Invalid edge."), QMessageBox::Ok, _widget);
							//box->exec();
						}
					}
					else if (_widget->get_ad2g().get_graph()->existEdge(_source, _tmpNode, true) != tlp::edge()) {
						QMessageBox* box = new QMessageBox(QMessageBox::Critical, tr("Edge already exist."), tr("This edge is already existing."), QMessageBox::Ok, _widget);
						box->exec();
					}
					else {
//						QMenu* menu = new QMenu;
//						QAction* directed = new QAction(tr("Directed"), menu);
//						QAction* undirected = new QAction(tr("Undirected"), menu);
//						menu->addAction(directed);
//						menu->addAction(undirected);
//						if (menu->exec(qMouseEv->globalPos()) == undirected) {
//							addLink();
//						}
						addLink();
					}
					_bends.clear();

					tlp::Observable::unholdObservers();
				}
				else {
				  tlp::Coord point(glMainWidget->width() - qMouseEv->x(), qMouseEv->y(), 0);
				  _bends.push_back(glMainWidget->getScene()->getCamera().screenTo3DWorld(point));
				}
			  }
			  glMainWidget->redraw();

			  return true;
		}


		// Abort edge tracing
		if (qMouseEv->buttons() == Qt::MidButton) {
			abort_edge_tracing();
			return true;
		}

		if (qMouseEv->button() == Qt::RightButton) {

			QMenu* menu = new QMenu;
			QAction* my_action;

			if (hoveringOverNode) {
				my_action = new QAction(tr("Remove view..."), menu);
				connect(my_action, SIGNAL(triggered()), _deleteNodeSignalMapper, SLOT(map()));
				_deleteNodeSignalMapper->setMapping(my_action, (int)_tmpNode);
			}
			else if (hoveringOverEdge) {
				my_action = new QAction(tr("Remove combining function..."), menu);
				connect(my_action, SIGNAL(triggered()), _deleteEdgeSignalMapper, SLOT(map()));
				_deleteEdgeSignalMapper->setMapping(my_action, (int)_tmpEdge);
			}
			else {
//				my_action = new QAction(tr("Add selected view"), menu);
//				connect(my_action, SIGNAL(triggered()), _addNodeSignalMapper, SLOT(map()));
//				_addNodeSignalMapper->setMapping(my_action, (QObject*) qMouseEv);
				return false;

			}

			menu->addAction(my_action);
			menu->exec(qMouseEv->globalPos());
			return true;
		}

		if  (e->type() == QEvent::MouseMove) {
			if (_translation_started && _tmpNode.isValid()) {
				mouse_translate(_tmpNode, qMouseEv->x(), qMouseEv->y(), glMainWidget);
			}
			else if (_edge_started) {
				tlp::Coord point(glMainWidget->width() - qMouseEv->x(), qMouseEv->y(), 0);
				point = glMainWidget->getScene()->getCamera().screenTo3DWorld(point);
				_curPos.set(point[0], point[1], point[2]);
				glMainWidget->redraw();
			}
			else {
				update_selection(graph, glMainWidget, hoveringOverNode, hoveringOverEdge);
			}

			return true;
		}
	}

	return false;
}

void PVGuiQt::AD2GInteractorComponent::update_selection(tlp::Graph* graph, tlp::GlMainWidget* glMainWidget, bool hoveringOverNode, bool hoveringOverEdge)
{
	std::string selectionPropertyName=glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getElementSelectedPropName();
	tlp::BooleanProperty* selection=graph->getProperty<tlp::BooleanProperty>(selectionPropertyName);

	if (hoveringOverNode) {
		if (!selection->getNodeValue(_tmpNode)) {
			selection->setAllEdgeValue(false);
			selection->setAllNodeValue(false);
			selection->setNodeValue(_tmpNode, true);
		}
	}
	else {
		if (hoveringOverEdge) {
			if (!selection->getEdgeValue(_tmpEdge)) {
				selection->setAllEdgeValue(false);
				selection->setAllNodeValue(false);
				selection->setEdgeValue(_tmpEdge, true);
			}
		}
		else {
			selection->setAllEdgeValue(false);
			selection->setAllNodeValue(false);
		}
	}
}

void PVGuiQt::AD2GInteractorComponent::abort_edge_tracing()
{
	_bends.clear();
	_edge_started = false;
	clearObserver();
	_glMainWidget->draw();
}

void PVGuiQt::AD2GInteractorComponent::addLink()
{
	tlp::edge newEdge = _widget->add_combining_function(_source, _tmpNode);

	tlp::Graph* graph = _glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getGraph();
	tlp::LayoutProperty* mLayout = graph->getProperty<tlp::LayoutProperty>(_glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getElementLayoutPropName());
	mLayout->setEdgeValue(newEdge, _bends);
}

bool PVGuiQt::AD2GInteractorComponent::draw(tlp::GlMainWidget* glMainWidget)
{
  if (!_edge_started) return false;

  glDisable(GL_STENCIL_TEST);
  glMainWidget->getScene()->getCamera().initGl();
  std::vector<tlp::Coord> lineVertices;
  lineVertices.push_back(_startPos);
  lineVertices.insert(lineVertices.end(), _bends.begin(), _bends.end());
  lineVertices.push_back(_curPos);
  std::vector<tlp::Color> lineColors;
  lineColors.resize(lineVertices.size(), tlp::Color(255,0,0,255));
  tlp::GlLine editedEdge(lineVertices, lineColors);
  editedEdge.draw(0,0);
  return true;
}

void PVGuiQt::AD2GInteractorComponent::mouse_translate(tlp::node /* n*/, double newX, double newY, tlp::GlMainWidget *glMainWidget) {

	if (newX - NODE_WIDTH < 0 || newX > glMainWidget->width() - NODE_WIDTH || newY  - NODE_HEIGHT< 0 || newY > glMainWidget->height() - NODE_HEIGHT) {
		return;
	}

	tlp::Observable::holdObservers();
	tlp::GlGraphInputData *inputData=glMainWidget->getScene()->getGlGraphComposite()->getInputData();
	tlp::Graph* graph = glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getGraph();
	tlp::LayoutProperty* layout = graph->getProperty<tlp::LayoutProperty>(glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getElementLayoutPropName());
	tlp::BooleanProperty* selection = graph->getProperty<tlp::BooleanProperty>(inputData->getElementSelectedPropName());
	tlp::Coord v0(0,0,0);
	tlp::Coord v1((double)(_editPosition[0] - newX), -(double)(_editPosition[1] - newY),0);


	v0 = glMainWidget->getScene()->getCamera().screenTo3DWorld(v0);
	v1 = glMainWidget->getScene()->getCamera().screenTo3DWorld(v1);
	v1 -= v0;
	tlp::Iterator<tlp::node> *itN = selection->getNodesEqualTo(true, graph);
	tlp::Iterator<tlp::edge> *itE = selection->getEdgesEqualTo(true, graph);
	layout->translate(v1, itN, itE);

	delete itN;
	delete itE;
	_editPosition[0]  = newX;
	_editPosition[1]  = newY;
	tlp::Observable::unholdObservers();
}

void PVGuiQt::AD2GInteractorComponent::initObserver(tlp::Graph *newGraph)
{
	newGraph->addGraphObserver(this);
	//graph = newGraph;
	_layoutProperty = newGraph->getProperty<tlp::LayoutProperty>("viewLayout");
	_layoutProperty->addPropertyObserver(this);
}

void PVGuiQt::AD2GInteractorComponent::clearObserver() {
	tlp::Graph* graph = _glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getGraph();
	if(graph) {
		graph->removeGraphObserver(this);
	}

	if(_layoutProperty) {
		_layoutProperty->removePropertyObserver(this);
	}

	_layoutProperty=NULL;
}

