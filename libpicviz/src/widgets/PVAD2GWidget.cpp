#include <picviz/widgets/PVAD2GWidget.h>

#include <tulip/Interactor.h>
#include <tulip/InteractorManager.h>
///
#include <tulip/TlpTools.h>
#include <tulip/TlpQtTools.h>
#include <tulip/PluginLoaderTxt.h>
#include <tulip/EdgeExtremityGlyphManager.h>
///

#include <picviz/PVView_types.h>
#include <picviz/PVView.h>

// Correlation
#include <pvkernel/core/PVAxisIndexType.h>
#include <picviz/PVCombiningFunctionView.h>
//#include <picviz/PVRFFAxesBind.h>
#include <picviz/PVTFViewRowFiltering.h>
#include <picviz/PVSelRowFilteringFunction.h>

#include <picviz/widgets/PVAD2GInteractor.h>


void PVWidgets::__impl::PVTableWidget::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		_dragStartPosition = event->pos();
	}

	QTableWidget::mousePressEvent(event);
}


void PVWidgets::__impl::PVTableWidget::mouseMoveEvent(QMouseEvent* event)
{
	if (!(event->buttons() & Qt::LeftButton))
		return;
	if ((event->pos() - _dragStartPosition).manhattanLength() < QApplication::startDragDistance())
		return;

	QTableWidgetItem* item = ((PVWidgets::PVAD2GWidget*) parent())->get_table()->currentItem();

	if (item->flags() & Qt::ItemIsEnabled) {
		QDrag* drag = new QDrag(this);
		QMimeData* mimeData = new QMimeData;
		void* ptr = item->data(Qt::UserRole).value<void*>();

		QByteArray byte_array;
		byte_array.reserve(sizeof(void*));
		byte_array.append((const char*)ptr, sizeof(void*));

		mimeData->setData("application/x-qabstractitemmodeldatalist", byte_array);

		drag->setMimeData(mimeData);

		Qt::DropAction dropAction = drag->exec(Qt::CopyAction | Qt::MoveAction);
	}

}

bool PVWidgets::__impl::FilterDropEvent::eventFilter(QObject* /*object*/, QEvent *event)
{
	if(event->type() == QEvent::DragEnter)
	{
		QDragEnterEvent* dragEnterEvent = static_cast<QDragEnterEvent*>(event);
		dragEnterEvent->accept(); // dragEnterEvent->acceptProposedAction();
		return true;
	}
	else if (event->type() == QEvent::Drop) {

		QDropEvent* dropEvent = static_cast<QDropEvent*>(event);

		if (dropEvent->mimeData()->hasFormat("application/x-qabstractitemmodeldatalist")) {
			dropEvent->acceptProposedAction();

			const QMimeData* mimeData = dropEvent->mimeData();
			QByteArray itemData = mimeData->data("application/x-qabstractitemmodeldatalist");

			if (itemData.size() < sizeof(Picviz::PVView**)) {
				return false;
			}
			Picviz::PVView* view = *(reinterpret_cast<Picviz::PVView* const*>(itemData.constData()));

			_widget->add_view(dropEvent->pos(), view);
<<<<<<< HEAD
			QString wn = view->get_window_name();

			// Disable QTableWidgetItem
			__impl::PVTableWidget* table = ((PVWidgets::PVAD2GWidget*) parent())->get_table();
			table->setCurrentCell(-1, -1);
			for (int i = 0; i < table->rowCount(); i++) {
				QTableWidgetItem* item = table->item(i, 0);
				if (item->data(Qt::UserRole).value<void*>() == (void*) view) {
					item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
				}
			}
=======
>>>>>>> b71d499b286a0904baac8c08d027b1f0588d81fc

			return true;
		}
	}

	return false;
}

PVWidgets::PVAD2GWidget::PVAD2GWidget(Picviz::PVAD2GView& ad2g, QMainWindow* mw /*= NULL*/) :
	QWidget(mw),
	_ad2g(ad2g),
	_mw(mw),
	_graph(_ad2g.get_graph())
{
	tlp::initTulipLib();
	tlp::PluginLoaderTxt txtPlug;
	//	tlp::loadPlugin("/usr/local/lib/tulip/libMixedModel-3.7.0.so", &txtPlug);
	tlp::loadPlugins(&txtPlug);   // library side plugins
	tlp::InteractorManager::getInst().loadPlugins(&txtPlug);
	tlp::GlyphManager::getInst().loadPlugins(&txtPlug);   // software side plugins, i.e. glyphs

	_nodeLinkView = new AD2GNodeLinkDiagramComponent();

	QHBoxLayout* main_layout = new QHBoxLayout();
	QWidget* nodeWidget = _nodeLinkView->construct(this);
	main_layout->addWidget(nodeWidget);
	_table = new __impl::PVTableWidget(this);
	main_layout->addWidget(_table);
	setLayout(main_layout);

	_nodeLinkView->hideOverview(true);

	tlp::DataSet dataSet;
	dataSet.set<bool>("arrow", true);
	dataSet.set<bool>("nodeLabel", true);
	dataSet.set<bool>("edgeLabel", false);

	_nodeLinkView->init();
	nodeWidget->setAcceptDrops(true);
	nodeWidget->installEventFilter(new __impl::FilterDropEvent(this));

	init_toolbar();
	fill_table();

	if (_graph) {
		openGraphOnGlMainWidget(_graph, &dataSet, _nodeLinkView->getGlMainWidget());

		// Set up graph rendering properties
		tlp::GlGraphRenderingParameters params = _nodeLinkView->getGlMainWidget()->getScene()->getGlGraphComposite()->getRenderingParameters();
		params.setEdgeColorInterpolate(false);
		params.setViewArrow(true);
		params.setAntialiasing(true);
		params.setViewNodeLabel(true);
		_nodeLinkView->getGlMainWidget()->getScene()->getGlGraphComposite()->setRenderingParameters(params);

		tlp::ColorProperty* color_property = _graph->getLocalProperty<tlp::ColorProperty>("viewColor");
		color_property->setAllNodeValue(tlp::Color(102, 0, 110));
		color_property->setAllEdgeValue(tlp::Color(142, 142, 142));
		_graph->getProperty<tlp::IntegerProperty>("viewFontSize")->setAllNodeValue(12);
		_graph->getLocalProperty<tlp::ColorProperty>("viewLabelColor")->setAllNodeValue(tlp::Color(255, 255, 255));
		//_graph->getProperty<tlp::IntegerProperty>("viewShape")->setAllEdgeValue(4);
		_graph->getProperty<tlp::IntegerProperty>("viewSrcAnchorShape")->setAllEdgeValue(-1);
		_graph->getProperty<tlp::IntegerProperty>("viewTgtAnchorShape")->setAllEdgeValue(50); // 28 for the Christmas Tree ! ;-)
		_graph->getProperty<tlp::SizeProperty>("viewTgtAnchorSize")->setAllEdgeValue(tlp::Size(0.5, 0.5, 0.5));

		initObservers();
	}

	_nodeLinkView->getGlMainWidget()->resizeGL(800,600);

	/////////////////////////////////////////////////////////////////////////////
	// Create hardcoded graph for testing purpose
//	Picviz::PVScene::list_views_t all_views = _ad2g.get_scene()->get_all_views();
//	Picviz::PVView_p view0 = all_views[0];
//	Picviz::PVView_p view1 = all_views[1];
//	tlp::node n0 = add_view(QPoint(100, 100), view0.get());
//	tlp::node n1 = add_view(QPoint(300, 300), view1.get());
//	add_combining_function(n0, n1);
	//////////////////////////////////////////////////////////////////////////////////

	// Apply layout algorithm:
//	tlp::LayoutProperty* viewLayout = _graph->getLocalProperty<tlp::LayoutProperty>("viewLayout");
//	std::string err;
//	_graph->applyPropertyAlgorithm("Mixed Model", viewLayout, err);
//	PVLOG_INFO("err=%s\n", err.c_str());
//	viewLayout->center(_graph);


}


PVWidgets::PVAD2GWidget::~PVAD2GWidget()
{
	clearObservers();
	delete _nodeLinkView;
}

void PVWidgets::PVAD2GWidget::init_toolbar()
{
	std::list<std::string> interactorsNamesAndPriorityMap(tlp::InteractorManager::getInst().getSortedCompatibleInteractors("Node Link Diagram view"));

	std::list<tlp::Interactor *> interactorsList;
	/*for (std::list<std::string>::reverse_iterator it = interactorsNamesAndPriorityMap.rbegin(); it != interactorsNamesAndPriorityMap.rend(); ++it) {
		interactorsList.push_back(tlp::InteractorManager::getInst().getInteractor((*it)));
	}*/
	AD2GInteractor* ad2g_interactor = new AD2GInteractor(this, _nodeLinkView->getGlMainWidget());
	interactorsList.push_back(ad2g_interactor);
	AD2GInteractor2* ad2g_interactor2 = new AD2GInteractor2(this, _nodeLinkView->getGlMainWidget());
	interactorsList.push_back(ad2g_interactor2);
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

void PVWidgets::PVAD2GWidget::change_interactor_slot()
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

void PVWidgets::PVAD2GWidget::add_view_Slot(QObject* mouse_event)
{
	QMouseEvent* event = (QMouseEvent*) mouse_event;
	Picviz::PVView* view = _ad2g.get_scene()->get_all_views()[_table->currentRow()].get();
	add_view(event->pos(), view);
}


<<<<<<< HEAD
void PVWidgets::PVAD2GWidget::add_view(QPoint pos, Picviz::PVView* view)
=======
tlp::node Picviz::PVAD2GWidget::add_view(QPoint pos, PVView* view)
>>>>>>> b71d499b286a0904baac8c08d027b1f0588d81fc
{
	tlp::Observable::holdObservers();

	// Add view to graph
	tlp::node newNode = _ad2g.add_view(view);

	// Compute view position
	tlp::Graph* graph = _nodeLinkView->getGlMainWidget()->getScene()->getGlGraphComposite()->getInputData()->getGraph();
	tlp::Coord point((double) _nodeLinkView->getGlMainWidget()->width() - (double) pos.x(),(double) pos.y(), 0);
	point = _nodeLinkView->getGlMainWidget()->getScene()->getCamera().screenTo3DWorld(point);
	tlp::Coord cameraDirection = _nodeLinkView->getGlMainWidget()->getScene()->getCamera().getEyes() - _nodeLinkView->getGlMainWidget()->getScene()->getCamera().getCenter();
	if(cameraDirection[0]==0 && cameraDirection[1]==0)
		point[2]=0;
	tlp::LayoutProperty* mLayout = graph->getProperty<tlp::LayoutProperty>(_nodeLinkView->getGlMainWidget()->getScene()->getGlGraphComposite()->getInputData()->getElementLayoutPropName());
	mLayout->setNodeValue(newNode, point);

	// Disable QTableWidgetItem
	int itemIndex = 0;
	_table->setCurrentCell(-1, -1);
	for (int i = 0; i < _table->rowCount(); i++) {
		QTableWidgetItem* item = _table->item(i, 0);
		if (item->data(Qt::UserRole).value<void*>() == (void*) view) {
			itemIndex = item->row();
			item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
		}
	}

	// Add node text
	tlp::StringProperty* label = graph->getProperty<tlp::StringProperty>("viewLabel");
	label->setNodeValue(newNode, qPrintable(QString::number(itemIndex+1)));

	tlp::Observable::unholdObservers();

	tlp::saveGraph(graph, "test.tlp");

	return newNode;
}

void PVWidgets::PVAD2GWidget::remove_view_Slot(int node)
{
	QMessageBox* box = new QMessageBox(QMessageBox::Question, tr("Confirm remove."), tr("Do you really want to remove this view?"), QMessageBox::Yes | QMessageBox::No, this);
	if (box->exec() == QMessageBox::Yes) {

		tlp::node n = (tlp::node) node;
		tlp::Graph* graph = _nodeLinkView->getGlMainWidget()->getScene()->getGlGraphComposite()->getInputData()->getGraph();
		tlp::Observable::holdObservers();

		// Enable item in table
		Picviz::PVView* view = _ad2g.get_view(n);
		for (int i = 0; i < _table->rowCount(); i++) {
			QTableWidgetItem* item = _table->item(i, 0);
			if (item->data(Qt::UserRole).value<void*>() == (void*) view) {
				item->setFlags(item->flags() | Qt::ItemIsEnabled);
			}
		}

		graph->delNode(n);
		_nodeLinkView->getGlMainWidget()->redraw();
		tlp::Observable::unholdObservers();
	}
}

tlp::edge PVWidgets::PVAD2GWidget::add_combining_function(const tlp::node source, const tlp::node target)
{
	tlp::Graph* graph = _nodeLinkView->getGlMainWidget()->getScene()->getGlGraphComposite()->getInputData()->getGraph();

	// RFF
	Picviz::PVCombiningFunctionView_p cf_sp(new Picviz::PVCombiningFunctionView());
	Picviz::PVTFViewRowFiltering* tf = cf_sp->get_first_tf();
	LIB_CLASS(Picviz::PVSelRowFilteringFunction) &row_filters = LIB_CLASS(Picviz::PVSelRowFilteringFunction)::get();
	Picviz::PVSelRowFilteringFunction_p rff_bind = row_filters.get_class_by_name("axes_bind");
	assert(rff_bind);
	rff_bind = rff_bind->clone<Picviz::PVSelRowFilteringFunction>();
	PVCore::PVArgumentList args;
	args["axis_org"].setValue(PVCore::PVAxisIndexType(1));
	args["axis_dst"].setValue(PVCore::PVAxisIndexType(1));
	rff_bind->set_args(args);
	tf->push_rff(rff_bind);
	Picviz::PVView* view_src = _ad2g.get_view(source);
	Picviz::PVView* view_dst = _ad2g.get_view(target);
	tlp::edge newEdge = _ad2g.set_edge_f(view_src, view_dst, cf_sp);

	_nodeLinkView->elementSelectedSlot(newEdge.id, false);

	return newEdge;
}

void PVWidgets::PVAD2GWidget::remove_combining_function_Slot(int edge)
{
	QMessageBox* box = new QMessageBox(QMessageBox::Question, tr("Confirm remove."), tr("Do you really want to remove this combining function?"), QMessageBox::Yes | QMessageBox::No, this);
	if (box->exec() == QMessageBox::Yes) {
		tlp::edge e = (tlp::edge) edge;

		tlp::Observable::holdObservers();
		tlp::Graph* graph = _nodeLinkView->getGlMainWidget()->getScene()->getGlGraphComposite()->getInputData()->getGraph();
		graph->delEdge(e);
		_nodeLinkView->getGlMainWidget()->redraw();
		Observable::unholdObservers();
	}
}

void PVWidgets::PVAD2GWidget::edit_combining_function(int edge)
{
	//PVWidgets::PVAD2GEdgeEditor* edge_editor = new PVWidgets::PVAD2GEdgeEditor();
	//edge_editor->exec();
}

void PVWidgets::PVAD2GWidget::initObservers()
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

void PVWidgets::PVAD2GWidget::highlightViewItem(tlp::node n)
{
	PVView* view = _ad2g.get_view(n);
	for (int i = 0; i < _table->rowCount(); i++) {
		QTableWidgetItem* item = _table->item(i, 0);
		item->setSelected(item->data(Qt::UserRole).value<void*>() == (void*) view && n != tlp::node());
	}
}

void PVWidgets::PVAD2GWidget::fill_table()
{
	_table->setColumnCount(1);

	Picviz::PVScene::list_views_t all_views = _ad2g.get_scene()->get_all_views();
	foreach (Picviz::PVView_p view, all_views) {
		QTableWidgetItem* item = new QTableWidgetItem(view->get_window_name());
		item->setData(Qt::UserRole, qVariantFromValue((void*) view.get()));
		_table->setRowCount(_table->rowCount()+1);
		_table->setItem(_table->rowCount()-1, 0, item);
	}

	_table->horizontalHeader()->hide();
	_table->horizontalHeader()->setStretchLastSection(true);
	_table->resizeRowsToContents();

	_table->setDragEnabled(true);
	_table->setDragDropMode(QAbstractItemView::DragOnly);
}

void PVWidgets::PVAD2GWidget::clearObservers()
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

void PVWidgets::PVAD2GWidget::update(ObserverIterator /*begin*/, ObserverIterator /*end*/)
{
	_nodeLinkView->draw();
}
