#include <picviz/widgets/PVAD2GWidget.h>

#include <tulip/Interactor.h>
#include <tulip/InteractorManager.h>

Picviz::PVAD2GWidget::PVAD2GWidget(PVAD2GView& ad2g, QMainWindow* mw /*= NULL*/) :
	QWidget(mw),
	_ad2g(ad2g),
	_mw(mw),
	_graph(_ad2g.get_graph())
{
	_nodeLinkView = new AD2GNodeLinkDiagramComponent();

	_widget = _nodeLinkView->construct(this);
	_nodeLinkView->hideOverview(true);

	tlp::DataSet dataSet;
	const QString& str = "tulip_graph.tlp";
	dataSet.set("file::filename", str);
	dataSet.set("arrow", true);

	if (_graph) {
		openGraphOnGlMainWidget(_graph, &dataSet, _nodeLinkView->getGlMainWidget());
		initObservers();
	}

	_nodeLinkView->init();
	init_toolbar();
}

Picviz::PVAD2GWidget::~PVAD2GWidget()
{
	clearObservers();
	delete _nodeLinkView;
}

QWidget* Picviz::PVAD2GWidget::get_widget()
{
	return _widget;
}

void Picviz::PVAD2GWidget::init_toolbar()
{
	//std::list<std::string> interactorsNamesAndPriorityMap(tlp::InteractorManager::getInst().getSortedCompatibleInteractors("Node Link Diagram view"));

	std::list<tlp::Interactor *> interactorsList;
	//for (std::list<std::string>::reverse_iterator it = interactorsNamesAndPriorityMap.rbegin(); it != interactorsNamesAndPriorityMap.rend(); ++it) {
	//	interactorsList.push_back(tlp::InteractorManager::getInst().getInteractor((*it)));
	//}
	AD2GInteractor* ad2g_interactor = new AD2GInteractor(this);
	interactorsList.push_back(ad2g_interactor);

	_nodeLinkView->setInteractors(interactorsList);
	_nodeLinkView->setActiveInteractor(ad2g_interactor);

//	_toolBar = new QToolBar(this);
//	std::list<QAction *> interactorsActionList;
//	for (std::list<tlp::Interactor *>::iterator it = interactorsList.begin(); it != interactorsList.end(); ++it)
//		interactorsActionList.push_back((*it)->getAction());
//	for (std::list<QAction *>::iterator it = interactorsActionList.begin(); it != interactorsActionList.end(); ++it) {
//		_toolBar->addAction(*it);
//		connect(*it, SIGNAL(triggered()), this, SLOT(change_interactor_slot()));
//	}
//
//	if (_mw) {
//		_mw->addToolBar(_toolBar);
//	}
}

void Picviz::PVAD2GWidget::change_interactor_slot()
{
	QAction *action = (QAction*) sender();
	QList<QAction*> actions = _toolBar->actions();

	for (QList<QAction*>::iterator it = actions.begin(); it != actions.end(); ++it) {
		(*it)->setChecked(false);
	}

	action->setCheckable(true);
	action->setChecked(true);

	tlp::InteractorAction *interactorAction = (tlp::InteractorAction *) action;
	_nodeLinkView->setActiveInteractor(interactorAction->getInteractor());
}

void Picviz::PVAD2GWidget::initObservers()
{
	if (_graph) {
		_graph->addObserver(this);
		tlp::Iterator<tlp::PropertyInterface*> *it = _graph->getObjectProperties();

		while (it->hasNext()) {
			tlp::PropertyInterface* tmp = it->next();
			tmp->addObserver(this);
		}

		delete it;
	}
}

void Picviz::PVAD2GWidget::clearObservers()
{
	if (_graph) {
		_graph->removeObserver(this);
		tlp::Iterator<tlp::PropertyInterface*> *it = _graph->getObjectProperties();

		while (it->hasNext()) {
			(it->next())->removeObserver(this);
		}

		delete it;
	}
}

void Picviz::PVAD2GWidget::update(ObserverIterator begin, ObserverIterator end)
{
	_nodeLinkView->draw();
}

bool Picviz::AD2GInteractorComponent::eventFilter(QObject* widget, QEvent* e)
{

	if (e->type() == QEvent::MouseButtonPress) {
		PVLOG_INFO("Picviz::AD2GInteractorComponent::eventFilter e->type()=%d\n", e->type());
		QMouseEvent* qMouseEv = (QMouseEvent *) e;
		tlp::GlMainWidget* glMainWidget = (tlp::GlMainWidget *) widget;

		if (qMouseEv->button()== Qt::RightButton) {
			// Enter here if we have released the left button of the mouse

			// doSelect function return node/edge under the mouse
			tlp::node tmpNode;
			tlp::edge tmpEdge;
			tlp::ElementType type;
			bool result = glMainWidget->doSelect(qMouseEv->x(), qMouseEv->y(), type, tmpNode, tmpEdge);

			if (result) {
				// Enter here if we have node/edge under the mouse

				// Store selection property
				tlp::Graph *graph=glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getGraph();
				std::string selectionPropertyName=glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getElementSelectedPropName();
				tlp::BooleanProperty* selection=graph->getProperty<tlp::BooleanProperty>(selectionPropertyName);

				// Before do anything on the graph, we push the current state of the graph (this activate the undo/redo system)
				graph->push();

				// Deselect all nodes/edges
				selection->setAllNodeValue(false);
				selection->setAllEdgeValue(false);

				switch(type) {
					case tlp::NODE:
					// Set selection at true for selected node
					selection->setNodeValue(tmpNode, true);
					break;

					case tlp::EDGE:
					// Set selection at false for selected edge
					selection->setEdgeValue(tmpEdge, true);
					break;
				}

				//QMessageBox box("Selection", type==tlp::NODE?QString().sprintf("You clicked on Node #%d", tmpNode.id):QString().sprintf("You clicked on Edge  #%d", tmpEdge.id), QMessageBox::Information, QMessageBox::Ok, QMessageBox::NoButton, QMessageBox::NoButton, (tlp::GlMainWidget *) widget);
				//box.exec();
				QMenu* menu = new QMenu;
				QAction* my_action = new QAction(tr("Add view"), menu);
				menu->addAction(my_action);
				menu->exec(qMouseEv->globalPos());

				// We have treated the event so we return true
				// (this event will not be passed to others interactorComponent)
				return true;
			}
		}
	}

   // We don't have treated the event so we return false
   // (this event will be passed to others interactorComponent)
   return false;
}
