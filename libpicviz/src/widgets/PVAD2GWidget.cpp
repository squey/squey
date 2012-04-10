#include <picviz/widgets/PVAD2GWidget.h>

#include <tulip/Interactor.h>
#include <tulip/InteractorManager.h>
///
#include <tulip/TlpTools.h>
#include <tulip/TlpQtTools.h>
#include <tulip/PluginLoaderTxt.h>
///

Picviz::PVAD2GWidget::PVAD2GWidget(PVAD2GView& ad2g, QMainWindow* mw /*= NULL*/) :
	QWidget(mw),
	_ad2g(ad2g),
	_mw(mw),
	_graph(_ad2g.get_graph())
{
	tlp::initTulipLib();
	tlp::PluginLoaderTxt txtPlug;
//	tlp::loadPlugin("/usr/local/lib/tulip/libMixedModel-3.7.0.so", &txtPlug);
    tlp::loadPlugins(&txtPlug);
	tlp::InteractorManager::getInst().loadPlugins(&txtPlug);
	tlp::GlyphManager::getInst().loadPlugins(&txtPlug);   // software side plugins, i.e. glyphs

	_nodeLinkView = new AD2GNodeLinkDiagramComponent();

	_widget = _nodeLinkView->construct(this);
	_nodeLinkView->hideOverview(true);

	_graph = tlp::loadGraph("/home/jbleonesio/tulip_graph_layout.tlp");



	tlp::DataSet dataSet;
	dataSet.set<bool>("arrow", true);

	_nodeLinkView->init();

	init_toolbar();

	if (_graph) {


		openGraphOnGlMainWidget(_graph, &dataSet, _nodeLinkView->getGlMainWidget());

		// Set up graph rendering properties
		tlp::GlGraphRenderingParameters params = _nodeLinkView->getGlMainWidget()->getScene()->getGlGraphComposite()->getRenderingParameters();
		params.setViewArrow(true);
		params.setEdgeColorInterpolate(false);

		tlp::ColorProperty* color_property = _graph->getLocalProperty<tlp::ColorProperty>("viewColor");
		color_property->setAllNodeValue(tlp::Color(102, 0, 110));
		color_property->setAllEdgeValue(tlp::Color(142, 142, 142));

		_nodeLinkView->getGlMainWidget()->getScene()->getGlGraphComposite()->setRenderingParameters(params);

		initObservers();
	}

	// Apply layout algorithm:
	tlp::LayoutProperty* viewLayout = _graph->getLocalProperty<tlp::LayoutProperty>("viewLayout");
	std::string err;
	_graph->applyPropertyAlgorithm("Mixed Model", viewLayout, err);
	PVLOG_INFO("err=%s\n", err.c_str());
	viewLayout->center(_graph);

	_nodeLinkView->getGlMainWidget()->resizeGL(800,600);
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
	std::list<std::string> interactorsNamesAndPriorityMap(tlp::InteractorManager::getInst().getSortedCompatibleInteractors("Node Link Diagram view"));

	std::list<tlp::Interactor *> interactorsList;
	for (std::list<std::string>::reverse_iterator it = interactorsNamesAndPriorityMap.rbegin(); it != interactorsNamesAndPriorityMap.rend(); ++it) {
		interactorsList.push_back(tlp::InteractorManager::getInst().getInteractor((*it)));
	}
	AD2GInteractor* ad2g_interactor = new AD2GInteractor(this, _nodeLinkView->getGlMainWidget());
	interactorsList.push_back(ad2g_interactor);

	_nodeLinkView->setInteractors(interactorsList);
	_nodeLinkView->setActiveInteractor(ad2g_interactor);

	_toolBar = new QToolBar(this);
	std::list<QAction *> interactorsActionList;
	for (std::list<tlp::Interactor *>::iterator it = interactorsList.begin(); it != interactorsList.end(); ++it)
		interactorsActionList.push_back((*it)->getAction());
	for (std::list<QAction *>::iterator it = interactorsActionList.begin(); it != interactorsActionList.end(); ++it) {
		_toolBar->addAction(*it);
		connect(*it, SIGNAL(triggered()), this, SLOT(change_interactor_slot()));
	}

	if (_mw) {
		_mw->addToolBar(_toolBar);
	}
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

Picviz::AD2GInteractorComponent::AD2GInteractorComponent(PVAD2GWidget* widget, tlp::GlMainWidget* glMainWidget, Qt::MouseButton button /*= Qt::LeftButton*/, Qt::KeyboardModifier modifier /*= Qt::NoModifier*/) :
	_widget(widget),
	_glMainWidget(glMainWidget)
{
	_deleteNodeSignalMapper = new QSignalMapper(this);
	connect(_deleteNodeSignalMapper, SIGNAL(mapped(int)), this, SLOT(remove_view_Slot(int)));

	_addNodeSignalMapper = new QSignalMapper(this);
	connect(_addNodeSignalMapper, SIGNAL(mapped(QObject*)), this, SLOT(add_view_Slot(QObject*)));
}

bool Picviz::AD2GInteractorComponent::eventFilter(QObject* widget, QEvent* e)
{
	QMouseEvent* qMouseEv = (QMouseEvent *) e;
	tlp::GlMainWidget* glMainWidget = (tlp::GlMainWidget *) widget;

	if (e->type() == QEvent::MouseMove) {
		tlp::node tmpNode;
		tlp::edge tmpEdge;
		tlp::ElementType type;
		bool result = glMainWidget->doSelect(qMouseEv->x(), qMouseEv->y(), type, tmpNode, tmpEdge);

		tlp::Graph *graph=glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getGraph();
		std::string selectionPropertyName=glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getElementSelectedPropName();
		tlp::BooleanProperty* selection=graph->getProperty<tlp::BooleanProperty>(selectionPropertyName);

		if (result) {
			switch(type) {
				case tlp::NODE:
					if (!selection->getNodeValue(tmpNode)){
						selection->setAllEdgeValue(false);
						selection->setAllNodeValue(false);
						selection->setNodeValue(tmpNode, true);
					}
				break;
				case tlp::EDGE:
					if (!selection->getEdgeValue(tmpEdge)){
						selection->setAllEdgeValue(false);
						selection->setAllNodeValue(false);
						selection->setEdgeValue(tmpEdge, true);
					}
				break;
			}
		}
		else
		{
			selection->setAllEdgeValue(false);
			selection->setAllNodeValue(false);
		}

		return true;
	}

	if (e->type() == QEvent::MouseButtonPress) {

		if (qMouseEv->button()== Qt::RightButton) {
			tlp::node tmpNode;
			tlp::edge tmpEdge;
			tlp::ElementType type;
			bool result = glMainWidget->doSelect(qMouseEv->x(), qMouseEv->y(), type, tmpNode, tmpEdge);

			QMenu* menu = new QMenu;
			QAction* my_action;

			if (result) {
				switch(type) {
					case tlp::NODE:
						my_action = new QAction(tr("Remove view"), menu);
						connect(my_action, SIGNAL(triggered()), _deleteNodeSignalMapper, SLOT(map()));
						_deleteNodeSignalMapper->setMapping(my_action, (int)tmpNode);
					break;
					case tlp::EDGE:
						my_action = new QAction(tr("Remove combining function"), menu);
					break;
				}
			}
			else {
				my_action = new QAction(tr("Add view"), menu);
				connect(my_action, SIGNAL(triggered()), _addNodeSignalMapper, SLOT(map()));
				_addNodeSignalMapper->setMapping(my_action, (QObject*) qMouseEv);
			}

			menu->addAction(my_action);
			menu->exec(qMouseEv->globalPos());
			return true;
		}
	}

   return false;
}

void Picviz::AD2GInteractorComponent::add_view_Slot(QObject* mouse_event)
{
	PVLOG_INFO("Picviz::AD2GInteractorComponent::add_view_Slot\n");

	tlp::Observable::holdObservers();

	QMouseEvent* qMouseEv = (QMouseEvent*) mouse_event;
	tlp::Graph* graph = _glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getGraph();
	tlp::node newNode = graph->addNode();

	tlp::Coord point((double) _glMainWidget->width() - (double) qMouseEv->x(),(double) qMouseEv->y(),0);
	point = _glMainWidget->getScene()->getCamera().screenTo3DWorld(point);
	tlp::Coord cameraDirection = _glMainWidget->getScene()->getCamera().getEyes() - _glMainWidget->getScene()->getCamera().getCenter();

	if(cameraDirection[0]==0 && cameraDirection[1]==0)
	  point[2]=0;

	tlp::LayoutProperty* mLayout = graph->getProperty<tlp::LayoutProperty>(_glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getElementLayoutPropName());
	mLayout->setNodeValue(newNode, point);

	tlp::Observable::unholdObservers();
}

void Picviz::AD2GInteractorComponent::remove_view_Slot(int node)
{
	PVLOG_INFO("Picviz::AD2GInteractorComponent::remove_view_Slot\n");
	tlp::node n = (tlp::node) node;
	tlp::Graph* graph=_glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getGraph();
	tlp::Observable::holdObservers();
	graph->delNode(n);
	tlp::Observable::unholdObservers();
}
