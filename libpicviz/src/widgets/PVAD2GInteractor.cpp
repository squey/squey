#include <picviz/widgets/PVAD2GInteractor.h>

#include <tulip/NodeLinkDiagramComponent.h>
#include <tulip/GlLine.h>


#include <picviz/widgets/PVAD2GWidget.h>

PVWidgets::AD2GInteractorComponent::AD2GInteractorComponent(PVAD2GWidget* widget, tlp::GlMainWidget* glMainWidget, Qt::MouseButton /*button*/ /*= Qt::LeftButton*/, Qt::KeyboardModifier /*modifier*/ /*= Qt::NoModifier*/) :
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

bool PVWidgets::AD2GInteractorComponent::eventFilter(QObject* widget, QEvent* e)
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
			abortEdgeTracing();
			return true;
		}
		else if (qKeyEvent->key() == Qt::Key_Control) {
			glMainWidget->setCursor(QCursor(Qt::PointingHandCursor));
		}
	}
	else if (e->type() == QEvent::KeyRelease) {
		QKeyEvent* qKeyEvent = (QKeyEvent*) e;
		if (qKeyEvent->key() == Qt::Key_Control) {
			glMainWidget->setCursor(QCursor(Qt::ArrowCursor));
		}
	}
	else {
		QMouseEvent* qMouseEv = (QMouseEvent*) e;

		bool hoveringOverNode = glMainWidget->doSelect(qMouseEv->x(), qMouseEv->y(), _type, _tmpNode, _tmpEdge) && _type == tlp::NODE;
		bool hoveringOverEdge = glMainWidget->doSelect(qMouseEv->x(), qMouseEv->y(), _type, _tmpNode, _tmpEdge) && _type == tlp::EDGE;

		// Edit graph layout
		if (qMouseEv->modifiers() & Qt::ControlModifier) {

			update_selection(graph, glMainWidget, hoveringOverNode, hoveringOverEdge);

			if (e->type() == QEvent::MouseButtonPress && qMouseEv->button()==Qt::LeftButton) {
				_translation_started = true;

				_editPosition[0] = qMouseEv->x();
				_editPosition[1] = qMouseEv->y();
			}
			else if (e->type() == QEvent::MouseMove)
			{
				if (_translation_started && _tmpNode.isValid()) {
					mMouseTranslate(_tmpNode, qMouseEv->x(), qMouseEv->y(), glMainWidget);
				}
			}
		}
		else {
			if (qMouseEv->button()==Qt::LeftButton) {
				if (qMouseEv->type() == QEvent::MouseButtonDblClick) {
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
	//							addLink(_widget, _tmpNode, _source);
	//						}
							addLink(_widget, _source, _tmpNode);
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
				abortEdgeTracing();
				return true;
			}

			if (qMouseEv->button() == Qt::RightButton) {

				QMenu* menu = new QMenu;
				QAction* my_action;

				if (hoveringOverNode) {
					my_action = new QAction(tr("Remove view"), menu);
					connect(my_action, SIGNAL(triggered()), _deleteNodeSignalMapper, SLOT(map()));
					_deleteNodeSignalMapper->setMapping(my_action, (int)_tmpNode);
				}
				else if (hoveringOverEdge) {
					my_action = new QAction(tr("Remove combining function"), menu);
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
				if (_edge_started) {
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
	}

	return false;
}

void PVWidgets::AD2GInteractorComponent::update_selection(tlp::Graph* graph, tlp::GlMainWidget* glMainWidget, bool hoveringOverNode, bool hoveringOverEdge)
{
	std::string selectionPropertyName=glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getElementSelectedPropName();
	tlp::BooleanProperty* selection=graph->getProperty<tlp::BooleanProperty>(selectionPropertyName);

	if (hoveringOverNode) {
		if (!selection->getNodeValue(_tmpNode)) {
			selection->setAllEdgeValue(false);
			selection->setAllNodeValue(false);
			selection->setNodeValue(_tmpNode, true);
			_widget->highlightViewItem(_tmpNode);
		}
	}
	else {
		_widget->highlightViewItem(tlp::node());
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

void PVWidgets::AD2GInteractorComponent::abortEdgeTracing()
{
	_bends.clear();
	_edge_started = false;
	clearObserver();
	_glMainWidget->draw();
}

void PVWidgets::AD2GInteractorComponent::addLink(QObject* /*widget*/, const tlp::node source, const tlp::node target)
{
	tlp::edge newEdge = _widget->add_combining_function(_source, _tmpNode);

	tlp::Graph* graph = _glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getGraph();
	tlp::LayoutProperty* mLayout = graph->getProperty<tlp::LayoutProperty>(_glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getElementLayoutPropName());
	mLayout->setEdgeValue(newEdge, _bends);
}

bool PVWidgets::AD2GInteractorComponent::draw(tlp::GlMainWidget* glMainWidget)
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

void PVWidgets::AD2GInteractorComponent::initObserver(tlp::Graph *newGraph)
{
	newGraph->addGraphObserver(this);
	//graph = newGraph;
	_layoutProperty = newGraph->getProperty<tlp::LayoutProperty>("viewLayout");
	_layoutProperty->addPropertyObserver(this);
}

void PVWidgets::AD2GInteractorComponent::clearObserver() {
	tlp::Graph* graph = _glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getGraph();
	if(graph) {
		graph->removeGraphObserver(this);
	}

	if(_layoutProperty) {
		_layoutProperty->removePropertyObserver(this);
	}

	_layoutProperty=NULL;
}

void PVWidgets::AD2GInteractorComponent::mMouseTranslate(tlp::node n, double newX, double newY, GlMainWidget *glMainWidget) {
  Observable::holdObservers();
  GlGraphInputData *inputData=glMainWidget->getScene()->getGlGraphComposite()->getInputData();
  tlp::Graph* graph = glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getGraph();
  tlp::LayoutProperty* layout = graph->getProperty<tlp::LayoutProperty>(glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getElementLayoutPropName());
  tlp::BooleanProperty* selection = graph->getProperty<tlp::BooleanProperty>(inputData->getElementSelectedPropName());
  Coord v0(0,0,0);
  Coord v1((double)(_editPosition[0] - newX), -(double)(_editPosition[1] - newY),0);
  v0 = glMainWidget->getScene()->getCamera().screenTo3DWorld(v0);
  v1 = glMainWidget->getScene()->getCamera().screenTo3DWorld(v1);
  v1 -= v0;
  Iterator<node> *itN = selection->getNodesEqualTo(true, graph);
  Iterator<edge> *itE = selection->getEdgesEqualTo(true, graph);
  layout->translate(v1, itN, itE);
  delete itN;
  delete itE;
  _editPosition[0]  = newX;
  _editPosition[1]  = newY;
  Observable::unholdObservers();
}

