//! \file PVXmlEditorWidget.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include <QSplitter>

#include <PVXmlEditorWidget.h>
#include <PVXmlTreeItemDelegate.h>
#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <PVXmlParamWidget.h>
#include <PVInputTypeMenuEntries.h>

#include <pvkernel/rush/PVSourceCreatorFactory.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/filter/PVFieldSplitterChunkMatch.h>

#define FORMAT_BUILDER_TITLE (QObject::tr("Format builder"))
/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::PVXmlEditorWidget
 *
 *****************************************************************************/
PVInspector::PVXmlEditorWidget::PVXmlEditorWidget(QWidget * parent):
	QDialog(parent)
{
	setWindowTitle(FORMAT_BUILDER_TITLE);
    
	QSplitter* main_splitter = new QSplitter(Qt::Vertical);
    /*
     * ****************************************************************************
     * Création of graphics elements.
     * ****************************************************************************
     */    
    QVBoxLayout *vb=new QVBoxLayout();
    vb->setMargin(0);
    QHBoxLayout *hb=new QHBoxLayout();
    vbParam=new QVBoxLayout();
    
    
    //initialisation of the toolbar.
    actionAllocation();
    initToolBar(vb);
    
    //initialisation of the splitters list
    initSplitters();

    menuBar =new QMenuBar();
    initMenuBar();
    vb->setMenuBar(menuBar);
    //layout()->setMenuBar(menuBar);
    
    vb->addItem(hb);
    
    
    //the view
    myTreeView = new PVXmlTreeView(this);
    hb->addWidget(myTreeView);

    
    //the model
    myTreeModel = new PVXmlDomModel(this);
    myTreeView->setModel(myTreeModel);

    
    hb->addItem(vbParam);
    //parameter board
    myParamBord_old_model = new PVXmlParamWidget(this);
    vbParam->addWidget(myParamBord_old_model);  

    //param board plugin splitter
    myParamBord = &emptyParamBoard;
    vbParam->addWidget(myParamBord);
    
	// Create a table for the preview of the NRAW
	_nraw_model = new PVNrawListingModel();
	_nraw_widget = new PVNrawListingWidget(_nraw_model);
	_nraw_widget->connect_preview(this, SLOT(slotExtractorPreview()));
	_nraw_widget->connect_axes_name(this, SLOT(set_axes_name_selected_row_Slot(int)));

	// Put the vb layout into a widget and add it to the splitter
	QWidget* vb_widget = new QWidget();
	vb_widget->setLayout(vb);
	main_splitter->addWidget(vb_widget);

	// Tab widget for the NRAW
	QTabWidget* nraw_tab = new QTabWidget();
	nraw_tab->addTab(_nraw_widget, tr("Normalization preview"));
	main_splitter->addWidget(nraw_tab);

	_inv_lines_widget = new QListWidget();
	nraw_tab->addTab(_inv_lines_widget, tr("Unmatched lines"));

	QVBoxLayout* main_layout = new QVBoxLayout();
	main_layout->addWidget(main_splitter);
    setLayout(main_layout);
    
    //setWindowModality(Qt::ApplicationModal);
    
    /*
     * ****************************************************************************
     * Initialisation de toutes les connexions.
     * ****************************************************************************
     */
    lastSplitterPluginAdding = -1;
    initConnexions();
    
	// AG: here, that's a bit tricky. We want our widget to have a maximize button,
	// but the only way to do that with Qt is to use setWindowFlag(Qt::Window).
	// According to Qt's documentation (and source code), as we are originally a widget,
	// this will set the pos of this window to absolute (0,0). That's not what we want,
	// because we want it centered, according to the main window's position (our parent).
	// So set the window flag and set the center os our gemotry to the center of our parent.
	QRect geom = QRect(0,0,700,500);
	setWindowFlags(Qt::Window);
	geom.moveCenter(parent->geometry().center());
	setGeometry(geom);
}
/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::~PVXmlEditorWidget
 *
 *****************************************************************************/
PVInspector::PVXmlEditorWidget::~PVXmlEditorWidget() {
    /*actionAddFilterAfter->deleteLater();
    actionAddRegExAfter->deleteLater();
    actionDelete->deleteLater();
    myTreeView->deleteLater();*/
}


/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::actionAllocation
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::actionAllocation(){
    actionAddAxisIn = new QAction("add an axis",(QObject*)this);
    actionAddAxisIn->setIcon(QIcon(":/add-axis"));
    actionAddFilterAfter = new QAction("add a filter",(QObject*)this);
    actionAddFilterAfter->setIcon(QIcon(":/filter"));
    actionAddRegExAfter = new QAction("add a RegEx",(QObject*)this);
    actionAddRegExAfter->setIcon(QIcon(":/add-regexp"));
    actionAddUrl = new QAction("add an URL",(QObject*)this);
    actionAddUrl->setIcon(QIcon(":/add-url"));

    actionSave = new QAction("&Save",(QObject*)this);
    actionSave->setShortcut(QKeySequence(Qt::CTRL+Qt::Key_S));
    actionSave->setIcon(QIcon(":/save"));
    actionSaveAs = new QAction("Save as...",(QObject*)this);
    actionDelete = new QAction("Delete",(QObject*)this);
    actionDelete->setShortcut(QKeySequence(Qt::Key_Delete));
    actionDelete->setIcon(QIcon(":/red-cross"));
    actionDelete->setEnabled(false);
    actionMoveDown = new QAction("move down",(QObject*)this);
    actionMoveDown->setShortcut(QKeySequence(Qt::Key_Down));
    actionMoveDown->setIcon(QIcon(":/go-down.png"));
    actionMoveUp = new QAction("move up", (QObject*) this);
    actionMoveUp->setShortcut(QKeySequence(Qt::Key_Up));
    actionMoveUp->setIcon(QIcon(":/go-up.png"));
    actionOpen = new QAction(tr("Open"),(QObject*)this);
    actionOpen->setShortcut(QKeySequence(Qt::CTRL+Qt::Key_O));
    actionOpen->setIcon(QIcon(":/document-open.png"));
}


/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::initConnexions
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::initConnexions() {
    //connexion to update the parameter board
    connect(myTreeView, SIGNAL(clicked(const QModelIndex &)), myParamBord_old_model, SLOT(edit(const QModelIndex &)));
    //connexion to endable/desable items in toolsbar menu.
    connect(myTreeView, SIGNAL(clicked(const QModelIndex &)), this, SLOT(slotUpdateToolDesabled(const QModelIndex &)));
    
    //data has changed from tree 
    connect(myTreeModel, SIGNAL(dataChanged(const QModelIndex &, const QModelIndex& )), myTreeView, SLOT(slotDataHasChanged(const QModelIndex & , const QModelIndex & )));
    
	// When an item is clicked in the tree view, auto-select the good axis in the mini-extractor
	connect(myTreeView, SIGNAL(clicked(const QModelIndex &)), this, SLOT(slotItemClickedInView(const QModelIndex &)));
	

    /*
     * Connexions for the toolBar.
     */
    connect(actionAddAxisIn,  SIGNAL(triggered()),this,SLOT(slotAddAxisIn()));
    connect(actionAddFilterAfter, SIGNAL(triggered()), this, SLOT(slotAddFilterAfter()));
    connect(actionAddRegExAfter, SIGNAL(triggered()), this, SLOT(slotAddRegExAfter()));
    connect(actionDelete, SIGNAL(triggered()), this, SLOT(slotDelete()));
    connect(actionMoveDown,SIGNAL(triggered()),this,SLOT(slotMoveDown()));
    connect(actionMoveUp,SIGNAL(triggered()),this,SLOT(slotMoveUp()));
    connect(actionOpen,SIGNAL(triggered()),this,SLOT(slotOpen()));
    connect(actionSave, SIGNAL(triggered()), this, SLOT(slotSave()));
    connect(actionSaveAs, SIGNAL(triggered()), this, SLOT(slotSaveAs()));
    connect(actionAddUrl, SIGNAL(triggered()), this, SLOT(slotAddUrl()));
    connect(myParamBord_old_model,SIGNAL(signalNeedApply()),this,SLOT(slotNeedApply()));
    connect(myParamBord_old_model,SIGNAL(signalSelectNext()),myTreeView,SLOT(slotSelectNext()));
    
    

}

/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::initToolBar
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::initToolBar(QVBoxLayout *vb){

    QToolBar *tools = new QToolBar();


    tools->addAction(actionAddFilterAfter);
    tools->addAction(actionAddAxisIn);

    tools->addSeparator();
    tools->addAction(actionMoveDown);
    tools->addAction(actionMoveUp);
    tools->addSeparator();
    tools->addAction(actionDelete);
    tools->addSeparator();
    tools->addAction(actionOpen);
    tools->addAction(actionSave);

    vb->addWidget(tools);
}
/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::initSplitters
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::initSplitters() {
        LIB_CLASS(PVFilter::PVFieldsSplitterParamWidget)::list_classes splitters = LIB_CLASS(PVFilter::PVFieldsSplitterParamWidget)::get().get_list();
        LIB_CLASS(PVFilter::PVFieldsFilterParamWidget<PVFilter::one_to_one>)::list_classes filters = LIB_CLASS(PVFilter::PVFieldsFilterParamWidget<PVFilter::one_to_one>)::get().get_list();

        // _list_* is a QHash. Its keys are a QString with the registered name of the class (in our case, "csv", "regexp", etc...).
        // Its values are a boost::shared_ptr<PVFieldsSplitterParamWidget> or boost::shared_ptr<PVFieldsFilterParamWidget<one_to_one> > object.
        // For instance :
        LIB_CLASS(PVFilter::PVFieldsSplitterParamWidget)::list_classes::const_iterator it;
        for (it = splitters.begin(); it != splitters.end(); it++) {
                PVFilter::PVFieldsSplitterParamWidget_p pluginsSplitter = it.value();
                assert(pluginsSplitter);
                _list_splitters.push_back(pluginsSplitter);
				pluginsSplitter->get_action_menu()->setData(it.key());
                connect(pluginsSplitter->get_action_menu(), SIGNAL(triggered()), this, SLOT(slotAddSplitter()));
        }
}


/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::slotAddAxisIn
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::slotAddAxisIn() {
    myTreeView->addAxisIn();
}

/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::slotAddFilterAfter
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::slotAddFilterAfter() {
    myTreeView->addFilterAfter();
}

/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::slotAddRegExAfter
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::slotAddRegExAfter() {
    myTreeView->addRegExIn();
}
/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::slotAddSplitter
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::slotAddSplitter()
{
        QAction* action_src = (QAction*) sender();
        QString const& itype = action_src->data().toString();
        PVFilter::PVFieldsSplitterParamWidget_p in_t = LIB_CLASS(PVFilter::PVFieldsSplitterParamWidget)::get().get_class_by_name(itype);
        PVFilter::PVFieldsSplitterParamWidget_p in_t_cpy = in_t->clone<PVFilter::PVFieldsSplitterParamWidget>();
		QString registered_name = in_t_cpy->registered_name();
        PVLOG_DEBUG("(PVInspector::PVXmlEditorWidget::slotAddSplitter) type_name %s, %s\n", qPrintable(in_t_cpy->type_name()), qPrintable(registered_name));
        myTreeView->addSplitter(in_t_cpy);
}


/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::slotAddUrl
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::slotAddUrl(){
    myTreeView->addUrlIn();
}


/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::slotApplyModification
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::slotApplyModification() {
  QModelIndex index;
    myTreeView->applyModification(myParamBord_old_model,index);
}


/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::slotDelete
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::slotDelete() {
    QDialog confirm(this);
    QVBoxLayout vb;
    confirm.setLayout(&vb);
    vb.addWidget(new QLabel("Do realy want to delete it ?"));
    QHBoxLayout bas;
    vb.addLayout(&bas);
    QPushButton no("No");
    bas.addWidget(&no);
    QPushButton yes("Yes");
    bas.addWidget(&yes);
    
    connect(&no,SIGNAL(clicked()),&confirm,SLOT(reject()));
    connect(&yes,SIGNAL(clicked()),&confirm,SLOT(accept()));

    //if confirmed then apply
    if(confirm.exec()){
        myTreeView->deleteSelection();
        QModelIndex ind;
        myParamBord_old_model->drawForNo(ind);
    }
}


/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::slotMoveUp
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::slotMoveUp() {
    myTreeView->moveUp();
}


/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::slotMoveDown
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::slotMoveDown() {
    myTreeView->moveDown();
}

/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::slotNeedApply
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::slotNeedApply(){
  QModelIndex index;
    myTreeView->applyModification(myParamBord_old_model,index);
}

/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::slotOpen
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::slotOpen() {
    QFileDialog fd;
    //open file chooser
    QString urlFile = fd.getOpenFileName(0, QString("Select the file."), PVRush::normalize_get_helpers_plugins_dirs(QString("text")).first());
	openFormat(urlFile);
}


/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::slotSave
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::slotSave() {
	if (_cur_file.isEmpty()) {
		slotSaveAs();
		return;
	}	

	if (myTreeModel->saveXml(_cur_file)) {
		return;
	}

	QMessageBox err(QMessageBox::Question, tr("Error while saving format"), tr("Unable to save the changes to %1. Do you want to save this format to another location ?").arg(_cur_file), QMessageBox::Yes | QMessageBox::No);
	if (err.exec() == QMessageBox::No) {
		return;
	}

	slotSaveAs();
}


/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::slotSaveAs
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::slotSaveAs() {
    QModelIndex index;
    myTreeView->applyModification(myParamBord_old_model,index);
    QFileDialog fd;
     //open file chooser
    QString urlFile = fd.getSaveFileName(0,QString("Select the file."),PVRush::normalize_get_helpers_plugins_dirs(QString("text")).first());
	if (!urlFile.isEmpty()) {
		if (myTreeModel->saveXml(urlFile)) {
			_cur_file = urlFile;
			setWindowTitleForFile(urlFile);
		}
	}
}

void PVInspector::PVXmlEditorWidget::setWindowTitleForFile(QString const& path)
{
	// Change the window title with the filename of the format
	QFileInfo fi(path);
	setWindowTitle(FORMAT_BUILDER_TITLE + QString(" - ") + fi.fileName());
}


/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::slotUpdateToolDesabled
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::slotUpdateToolDesabled(const QModelIndex &index){
    PVRush::PVXmlTreeNodeDom *node = myTreeModel->nodeFromIndex(index);
    
    //hideParamBoard();
    
    if (node->getDom().tagName() == "field") {
        myTreeView->expandRecursive(index);
        actionAddFilterAfter->setEnabled(true);
        actionAddAxisIn->setEnabled(true);
        actionAddRegExAfter->setEnabled(true);
        actionAddUrl->setEnabled(true);
        actionDelete->setEnabled(false);
    } else if (node->getDom().tagName() == "axis") {
        actionAddFilterAfter->setEnabled(false);
        actionAddAxisIn->setEnabled(false);
        actionAddRegExAfter->setEnabled(false);
        actionAddUrl->setEnabled(false);
        actionDelete->setEnabled(true);
    } else if (node->getDom().tagName() == "filter") {
        actionAddFilterAfter->setEnabled(false);
        actionAddAxisIn->setEnabled(false);
        actionAddRegExAfter->setEnabled(false);
        actionAddUrl->setEnabled(false);
        actionDelete->setEnabled(true);
    } else if (node->getDom().tagName() == "splitter") {
        myTreeView->expandRecursive(index);
        actionAddFilterAfter->setEnabled(false);
        actionAddAxisIn->setEnabled(false);
        actionAddRegExAfter->setEnabled(false);
        actionAddUrl->setEnabled(false);
        actionDelete->setEnabled(true);
    } else if (node->getDom().tagName() == "RegEx") {
        myTreeView->expandRecursive(index);
        actionAddFilterAfter->setEnabled(false);
        actionAddAxisIn->setEnabled(false);
        actionAddRegExAfter->setEnabled(false);
        actionAddUrl->setEnabled(false);
        actionDelete->setEnabled(true);
    } else if (node->getDom().tagName() == "url") {
        actionAddFilterAfter->setEnabled(false);
        actionAddAxisIn->setEnabled(false);
        actionAddRegExAfter->setEnabled(false);
        actionAddUrl->setEnabled(false);
        actionDelete->setEnabled(true);
    } else {
        actionAddFilterAfter->setEnabled(true);
        actionAddAxisIn->setEnabled(true);
        actionAddRegExAfter->setEnabled(true);
        actionAddUrl->setEnabled(true);
        actionDelete->setEnabled(false);
    }
}



/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::initMenuBar
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::initMenuBar() {
        QMenu *file = menuBar->addMenu(tr("&File"));
        file->addAction(actionOpen);
        file->addAction(actionSave);
        file->addAction(actionSaveAs);
        file->addSeparator();
		PVInputTypeMenuEntries::add_inputs_to_menu(file, this, SLOT(slotOpenLog()));
        file->addSeparator();


        QMenu *splitter = menuBar->addMenu(tr("&Splitter"));
        splitter->addAction(actionAddUrl);  
        splitter->addSeparator();
        //add all plugins splitters
        for (int i = 0; i < _list_splitters.size(); i++) {
                QAction *action = _list_splitters.at(i)->get_action_menu();
                assert(action);
                if (action) {
                        splitter->addAction(action);
                }
        }
}


/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::slotOpenLog
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::slotOpenLog()
{
	PVRush::PVInputType_p in_t = PVInputTypeMenuEntries::input_type_from_action((QAction*) sender());
	PVRush::list_creators lcr = PVRush::PVSourceCreatorFactory::get_by_input_type(in_t);

	QString choosenFormat;
	PVRush::PVInputType::list_inputs inputs;
	PVRush::hash_formats formats;

	if (!in_t->createWidget(formats, inputs, choosenFormat, this))
		return; // This means that the user pressed the "cancel" button

	_nraw_model->set_consistent(false);
	try {
		// Get the first input selected
		_log_input = inputs.front();
		PVLOG_DEBUG("Input: %s\n", qPrintable(in_t->human_name_of_input(_log_input)));

		// Pre discover the input w/ the source creators
		PVRush::list_creators::const_iterator itcr;
		_log_sc.reset();
		_log_input_type.reset();
		for (itcr = lcr.begin(); itcr != lcr.end(); itcr++) {
			PVRush::PVSourceCreator_p sc = *itcr;
			if (sc->pre_discovery(_log_input)) {
				_log_sc = sc;
				break;
			}
		}

		if (!_log_sc) {
			_log_input = PVCore::PVArgument(); // No log input
			QMessageBox box(QMessageBox::Critical, tr("Error"), tr("No input plugins can manage the source file '%1'. Aborting...").arg(in_t->human_name_of_input(_log_input)));
			box.show();
			return;
		}

		_log_input_type = in_t;

		// First extraction
		create_extractor();
		if (is_dom_empty()) {
			guess_first_splitter();
		}

		update_table(FORMATBUILDER_EXTRACT_START_DEFAULT, FORMATBUILDER_EXTRACT_END_DEFAULT);

	}
	catch (PVRush::PVInputException &e) {
		_log_input = PVCore::PVArgument(); // No log input
		QMessageBox err(QMessageBox::Critical, tr("Error"), tr("Error while importing a source: %1").arg(QString(e.what().c_str())));
		err.show();
		return;
	}

	if (!_nraw_model->is_consistent()) {
		_nraw_model->set_consistent(true);
	}

	// Tell the NRAW widget that the input has changed
	_nraw_widget->set_last_input(_log_input_type, _log_input);
	_nraw_widget->resize_columns_content();
}

void PVInspector::PVXmlEditorWidget::create_extractor()
{
	if (_log_extract) {
		_log_extract->force_stop_controller();
	}
	_log_extract.reset(new PVRush::PVExtractor());
	_log_extract->dump_elts(true);
	_log_extract->start_controller();
	// The moni-extractor use the discovery source, as not that much processing is done (it can be handle locally for instance !)
	_log_source = _log_sc->create_discovery_source_from_input(_log_input);
	_log_extract->add_source(_log_source);
}

void PVInspector::PVXmlEditorWidget::guess_first_splitter()
{
	// Guess first splitter and add it to the dom before parsing it !
	// The dom is the reference in here.

	PVLOG_DEBUG("(format_builder) trying to guess first splitter...");
	PVCol naxes;
	PVFilter::PVFieldsSplitter_p sp = PVFilter::PVFieldSplitterChunkMatch::get_match_on_input(_log_source, naxes);
	if (!sp) {
		// No splitter matches, just do nothing
		return;
	}

	// Ok, we got a match, add it to the dom.
	QString first_input_name = _log_input_type->human_name_of_input(_log_input);
	PVLOG_INFO("(format_builder) For input '%s', found a splitter that creates %d axes. Arguments:\n", qPrintable(first_input_name), naxes);
	PVCore::dump_argument_list(sp->get_args());

	QString msg = tr("It appears that the %1 splitter can process '%2' and create %3 fields.\n").arg(sp->registered_name()).arg(first_input_name).arg(naxes);
	msg += tr("Do you want to automatically add that splitter to the format ?");
	QMessageBox ask_auto(QMessageBox::Question, tr("Filter automatically found"), msg, QMessageBox::Yes | QMessageBox::No, this);
	if (ask_auto.exec() == QMessageBox::No) {
		return;
	}
	
	// Get the widget that comes with the splitter. TODO: do better than that
	QString type_name = sp->type_name();
	QString filter_name = sp->registered_name();
	PVFilter::PVFieldsSplitterParamWidget_p sp_widget = LIB_CLASS(PVFilter::PVFieldsSplitterParamWidget)::get().get_class_by_name(filter_name);
	if (!sp_widget) {
		PVLOG_WARN("Filter '%s' of type '%s' has no associated widget !\n", qPrintable(type_name), qPrintable(filter_name));
		return;
	}

	// Then we need to create 'naxes' children
	QStringList axes_name;
	for (PVCol i = 0; i < naxes; i++) {
		axes_name << QString("Axis %1").arg(i+1);
	}

	PVRush::PVXmlTreeNodeDom* node = myTreeModel->addSplitterWithAxes(myTreeModel->index(0,0,QModelIndex()), sp_widget, axes_name);
	node->setFromArgumentList(sp->get_args());
}

/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::hideParamBoard
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::hideParamBoard(){
        PVLOG_DEBUG("PVInspector::PVXmlEditorWidget::hideParamBoard\n");
}

void PVInspector::PVXmlEditorWidget::set_format_from_dom()
{
	QDomElement const& rootDom = myTreeModel->getRootDom();
	PVRush::PVFormat format;
	format.dump_elts(true);
	format.populate_from_xml(rootDom, true);
	_log_extract->set_format(format);
	_log_extract->set_chunk_filter(_log_extract->get_format().create_tbb_filters());
}


/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::showParamBoard
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::showParamBoard(PVRush::PVXmlTreeNodeDom *node){
        assert(node);
        PVLOG_DEBUG("PVInspector::PVXmlEditorWidget::showParamBoard()\n");
        
        //myParamBord=node->getSplitterPlugin()->get_param_widget();

}

void PVInspector::PVXmlEditorWidget::update_table(PVRow start, PVRow end)
{
	if (!_log_extract) {
		return;
	}

	assert(end > start);
	if (_nraw_model->is_consistent()) {
		_nraw_model->set_consistent(false);
	}

	// Here, two extractions are made.
	// The first one use the aggregator of the extract to get the data through
	// the filters of the widget (so that they can populate themselves).
	// Then, the format is created according to the DOM, the real extraction is
	// made and we get back the invalid elements.
	// AG: this is clearly subefficient but this is what I can do w/ the time I have.
	
	// First extraction
	
	// Clear the filter previous data
	myTreeModel->clearFiltersData();

	// Get the aggregator
	PVRush::PVAggregator& agg = _log_extract->get_agg();
	agg.set_strict_mode(true);
	agg.process_indexes(start, end);
	// And push the output through our filter tree
	PVCore::PVChunk* ck = agg();
	size_t nelts = 0;
	while (ck) {
		PVCore::list_elts::const_iterator it_elt;
		for (it_elt = ck->c_elements().begin(); it_elt != ck->c_elements().end(); it_elt++) {
			// The first field of a freshly created element is the whole element itself
			myTreeModel->processChildrenWithField((*it_elt)->c_fields().front());
			nelts++;
		}
		ck = agg();
	}

	// Update the data displaying of the filter param widgers
	myTreeModel->updateFiltersDataDisplay();

	// Do the real extraction using the DOM we just updated
	set_format_from_dom();
	// Create the nraw thanks to the extractor
	PVRush::PVControllerJob_p job = _log_extract->process_from_agg_idxes(start, end);
	job->wait_end();
	_log_extract->dump_nraw();
	_nraw_model->set_nraw(_log_extract->get_nraw());

	_nraw_model->set_consistent(true);

	// Set the invalid lines widget
	_inv_lines_widget->clear();
	QStringList& elts_invalid = job->get_invalids_elts();
	QStringList::iterator it_ie;
	for (it_ie = elts_invalid.begin(); it_ie != elts_invalid.end(); it_ie++) {
		QString const& line = *it_ie;
		_inv_lines_widget->addItem(line);
	}
}

void PVInspector::PVXmlEditorWidget::slotExtractorPreview()
{
	PVRow start,end;
	_nraw_widget->get_ext_args(start,end);
	update_table(start,end);
}

bool PVInspector::PVXmlEditorWidget::is_dom_empty()
{
	QDomElement const& rootDom = myTreeModel->getRootDom();
	return !rootDom.hasChildNodes();
}

void PVInspector::PVXmlEditorWidget::slotItemClickedInView(const QModelIndex &index)
{
	// Automatically set the good columns in the mini-extractor
	
	// Get the PVXmlTreeNodeDom object that comes with that index
    PVRush::PVXmlTreeNodeDom *node = myTreeModel->nodeFromIndex(index);

	// If this is the root item, do nothing.
	if (!node || node->getParent() == NULL) {
		_nraw_widget->unselect_column();
		return;
	}
	
	// Then, update the linear fields id in PVXmlTreeNode's tree.
	myTreeModel->updateFieldsLinearId();
	
	// If this is not a field, get the parent field
	if (node->typeToString() != "field" || node->getFieldLinearId() == -1) {
		node = node->getFirstFieldParent();
		// If it can't find any field parent, just return.
		// (but this is weird, that should not happen)
		if (!node) {
			_nraw_widget->unselect_column();
			return;
		}
	}
	
	// Then get that field's linear id
	PVCol field_id = node->getFieldLinearId();
	
	// And tell that to the mini-extractor widget
	_nraw_widget->select_column(field_id);
}

void PVInspector::PVXmlEditorWidget::set_axes_name_selected_row_Slot(int row)
{
	PVRush::PVNraw const& nraw = _log_extract->get_nraw();
	// We could use QList::fromVector(QVector::fromStdVector(nraw_table_line)), but that's not really efficient...
	if (row >= nraw.get_table().size()) {
		PVLOG_WARN("(PVXmlEditorWidget::set_axes_name_selected_row_Slot) row index '%d' does not exist in the current NRAW (size '%d').\n", row, nraw.get_table().size());
		return;
	}
	QStringList names;
	PVRush::PVNraw::nraw_table_line const& line = nraw.get_table().at(row);
	PVRush::PVNraw::nraw_table_line::const_iterator it;
	for (it = line.begin(); it != line.end(); it++) {
		names << *it;
	}
	myTreeModel->setAxesNames(names);
}

void PVInspector::PVXmlEditorWidget::set_axes_type_selected_row_Slot(int row)
{
}

void PVInspector::PVXmlEditorWidget::openFormat(QString const& path)
{
    QFile f(path);
    if (f.exists()) {//if the file exists...
        if (myTreeModel->openXml(path)) {
			_cur_file = path;
			setWindowTitleForFile(path);
		}
    }
}

void PVInspector::PVXmlEditorWidget::openFormat(QDomDocument& doc)
{
	myTreeModel->openXml(doc);
}
