#include <picviz/widgets/PVAD2GWidget.h>

#include <tulip/Interactor.h>
#include <tulip/InteractorManager.h>
#include <picviz/widgets/PVAD2GInteractor.h>
#include <picviz/widgets/PVAD2GListEdgesWidget.h>

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

	if (item && (item->flags() & Qt::ItemIsEnabled)) {
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

			return true;
		}
	}
	////
	else if (event->type() == QEvent::KeyPress) {
		QKeyEvent* qKeyEvent = (QKeyEvent*) event;
		if (qKeyEvent->key() == Qt::Key_AltGr) {
			PVAD2GWidget* widget = (PVAD2GWidget*) parent();
			widget->set_edit_graph(false);
		}
	}
	else if (event->type() == QEvent::KeyRelease) {
		QKeyEvent* qKeyEvent = (QKeyEvent*) event;
		if (qKeyEvent->key() == Qt::Key_AltGr) {
			PVAD2GWidget* widget = (PVAD2GWidget*) parent();
			widget->set_edit_graph(true);
		}
	}

	return false;
}

PVWidgets::PVAD2GWidget::PVAD2GWidget(Picviz::PVAD2GView& ad2g, QWidget* parent):
	QWidget(parent),
	_ad2g(ad2g),
	_graph(_ad2g.get_graph())
{
	tlp::initTulipLib();

	// Widgets
	_nodeLinkView = new AD2GNodeLinkDiagramComponent();
	QWidget* nodeWidget = _nodeLinkView->construct(this);
	_table = new __impl::PVTableWidget(this);
	_list_edges_widget = new PVAD2GListEdgesWidget(_ad2g, this);

	QHBoxLayout* radio_layout = new QHBoxLayout();
	_radio_edit_graph = new QRadioButton(tr("Edit graph"));
	_radio_edit_graph->setChecked(true);
	_radio_edit_layout = new QRadioButton(tr("Edit layout"));
	_radio_edit_layout->setToolTip(tr("Use AltGr to edit graph layout"));
	radio_layout->addWidget(_radio_edit_graph);
	radio_layout->addWidget(_radio_edit_layout);
	radio_layout->addStretch(1);

	connect(_radio_edit_graph, SIGNAL(toggled(bool)), this, SLOT(update_interactor_Slot()));
	connect(_radio_edit_layout, SIGNAL(toggled(bool)), this, SLOT(update_interactor_Slot()));

	// Layout
	QHBoxLayout* graph_views_layout = new QHBoxLayout();
	QVBoxLayout* graph_radio_layout = new QVBoxLayout();
	graph_radio_layout->addWidget(nodeWidget);
	graph_radio_layout->addLayout(radio_layout);
	graph_views_layout->addLayout(graph_radio_layout);
	graph_views_layout->addWidget(_table);
	QHBoxLayout* edges_properties_layout = new QHBoxLayout();
	edges_properties_layout->addWidget(_list_edges_widget);
	QVBoxLayout* main_layout = new QVBoxLayout();
	main_layout->addLayout(graph_views_layout);
	main_layout->addLayout(radio_layout);
	main_layout->addLayout(edges_properties_layout);
	setLayout(main_layout);

	_ad2g_interactor = new AD2GInteractor(this, _nodeLinkView->getGlMainWidget());
	_ad2g_interactor2 = new AD2GInteractor2(this, _nodeLinkView->getGlMainWidget());
	_nodeLinkView->setActiveInteractor(_ad2g_interactor);

	_nodeLinkView->hideOverview(true);

	tlp::DataSet dataSet;
	dataSet.set<bool>("arrow", true);
	dataSet.set<bool>("nodeLabel", true);
	dataSet.set<bool>("edgeLabel", false);

	_nodeLinkView->init();
	nodeWidget->setAcceptDrops(true);
	nodeWidget->installEventFilter(new __impl::FilterDropEvent(this));

	init_table();
	update_list_views();

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
}

void PVWidgets::PVAD2GWidget::set_edit_graph(bool edit_graph)
{
	if (edit_graph) {
		_radio_edit_graph->setChecked(true);
	}
	else {
		_radio_edit_layout->setChecked(true);
	}
}

void PVWidgets::PVAD2GWidget::update_interactor_Slot()
{
	if (_radio_edit_graph->isChecked()) {
		_nodeLinkView->setActiveInteractor(_ad2g_interactor);
	}
	else {
		_nodeLinkView->setActiveInteractor(_ad2g_interactor2);
	}
}


PVWidgets::PVAD2GWidget::~PVAD2GWidget()
{
	clearObservers();
	delete _nodeLinkView;
}

void PVWidgets::PVAD2GWidget::add_view_Slot(QObject* mouse_event)
{
	QMouseEvent* event = (QMouseEvent*) mouse_event;
	Picviz::PVView* view = _ad2g.get_scene()->get_all_views()[_table->currentRow()].get();
	add_view(event->pos(), view);
}


tlp::node PVWidgets::PVAD2GWidget::add_view(QPoint pos, Picviz::PVView* view)
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
	label->setNodeValue(newNode, qPrintable(QString::number(view->get_display_view_id())));

	// Set view id property
	graph->getProperty<tlp::IntegerProperty>("view_id")->setNodeValue(newNode, view->get_display_view_id());

	tlp::Observable::unholdObservers();

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

	Picviz::PVCombiningFunctionView_p cf_sp(new Picviz::PVCombiningFunctionView());
	tlp::edge newEdge = _ad2g.set_edge_f(source, target, cf_sp);

	_list_edges_widget->update_list_edges();

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

		_list_edges_widget->update_list_edges();

		_nodeLinkView->getGlMainWidget()->redraw();
		Observable::unholdObservers();
	}
}

/*void PVWidgets::PVAD2GWidget::select_edge(Picviz::PVView* view_src, Picviz::PVView* view_dst)
{
	tlp::node src = _ad2g.get_graph_node(view_src);
	tlp::node dst = _ad2g.get_graph_node(view_dst);
	tlp::edge edge = _graph->existEdge(src, dst);
	tlp::Graph* graph = _nodeLinkView->getGlMainWidget()->getScene()->getGlGraphComposite()->getInputData()->getGraph();
	graph->getProperty<tlp::ColorProperty>("viewColor")->setEdgeValue(edge, tlp::Color(255, 0, 0));
}*/

void PVWidgets::PVAD2GWidget::edit_combining_function(tlp::edge edge, tlp::node src, tlp::node dst)
{

	Picviz::PVView* view_src = _ad2g.get_view(src);
	Picviz::PVView* view_dst = _ad2g.get_view(dst);
	Picviz::PVCombiningFunctionView_p combining_function = _ad2g.get_edge_f(edge);

	tlp::Graph* graph = _nodeLinkView->getGlMainWidget()->getScene()->getGlGraphComposite()->getInputData()->getGraph();
	tlp::IntegerProperty* view_id_property = graph->getProperty<tlp::IntegerProperty>("view_id");
	int src_view_id = view_id_property->getNodeValue(src);
	int dst_view_id = view_id_property->getNodeValue(dst);

	_list_edges_widget->select_row(src_view_id, dst_view_id);

	_ad2g.set_selected_edge(view_src, view_dst);

	//_edge_editor->update(*view_src, *view_dst, *combining_function);
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
	Picviz::PVView* view = _ad2g.get_view(n);
	for (int i = 0; i < _table->rowCount(); i++) {
		QTableWidgetItem* item = _table->item(i, 0);
		item->setSelected(item->data(Qt::UserRole).value<void*>() == (void*) view && n != tlp::node());
	}
}

void PVWidgets::PVAD2GWidget::init_table()
{
	_table->setColumnCount(1);

	_table->horizontalHeader()->hide();
	_table->horizontalHeader()->setStretchLastSection(true);

	_table->setDragEnabled(true);
	_table->setDragDropMode(QAbstractItemView::DragOnly);
}

void PVWidgets::PVAD2GWidget::update_list_views()
{
	_table->clear();
	_table->setRowCount(0);

	Picviz::PVScene::list_views_t all_views = _ad2g.get_scene()->get_all_views();
	_table->setRowCount(all_views.count());
	foreach (Picviz::PVView_p view, all_views) {
		QTableWidgetItem* item = new QTableWidgetItem(view->get_source_parent()->get_name());
		item->setToolTip(view->get_window_name());
		item->setData(Qt::UserRole, qVariantFromValue((void*) view.get()));
		_table->setItem(view->get_view_id(), 0, item);
	}
	_table->resizeRowsToContents();
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

void PVWidgets::PVAD2GWidget::update_list_edges()
{
	if (_list_edges_widget) {
		_list_edges_widget->update_list_edges();
	}
}
