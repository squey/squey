//! \file PVXmlEditorWidget.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011


#include <PVXmlEditorWidget.h>
#include <PVXmlTreeItemDelegate.h>
#include <pvrush/PVXmlTreeNodeDom.h>
#include <PVXmlParamWidget.h>
#include <PVInputTypeMenuEntries.h>

#include <pvrush/PVSourceCreatorFactory.h>
#include <pvfilter/PVFieldSplitterChunkMatch.h>

/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::PVXmlEditorWidget
 *
 *****************************************************************************/
PVInspector::PVXmlEditorWidget::PVXmlEditorWidget(QWidget * parent):QWidget(parent) {
        
    setObjectName("PVXmlEditorWidget");
    
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
    myParamBord_old_model = new PVXmlParamWidget();
    vbParam->addWidget(myParamBord_old_model);  

    //param board plugin splitter
    myParamBord = &emptyParamBoard;
    vbParam->addWidget(myParamBord);
    
	// Create a table for the preview of the NRAW
	_nraw_model = new PVNrawListingModel();
	_nraw_widget = new PVNrawListingWidget(_nraw_model);
	_nraw_widget->connect_preview(this, SLOT(slotExtractorPreview()));

	// Tab widget for the NRAW
	QTabWidget* nraw_tab = new QTabWidget();
	nraw_tab->addTab(_nraw_widget, tr("Normalisation preview"));
	vb->addWidget(nraw_tab);

    setLayout(vb);
    
    //setWindowModality(Qt::ApplicationModal);
    
    /*
     * ****************************************************************************
     * Initialisation de toutes les connexions.
     * ****************************************************************************
     */
    lastSplitterPluginAdding = -1;
    initConnexions();
    
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
        int cpt=0;
        for (it = splitters.begin(); it != splitters.end(); it++) {
                PVFilter::PVFieldsSplitterParamWidget_p pluginsSplitter = it.value();
                assert(pluginsSplitter);
                pluginsSplitter->set_id(cpt);
                pluginsSplitter->get_action_menu()->setData(QVariant(it.key()));
                _list_splitters.insert(cpt++,pluginsSplitter);
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
void PVInspector::PVXmlEditorWidget::slotAddSplitter() {
        PVLOG_DEBUG("PVInspector::PVXmlEditorWidget::slotAddSplitter() \n");
        QAction* action_src = (QAction*) sender();
        QString const& itype = action_src->data().toString();
        PVLOG_DEBUG("sender():%s\n",action_src->iconText().toStdString().c_str());
        PVFilter::PVFieldsSplitterParamWidget_p in_t = LIB_CLASS(PVFilter::PVFieldsSplitterParamWidget)::get().get_class_by_name(itype);
        PVFilter::PVFieldsSplitterParamWidget_p in_t_cpy = in_t->clone<PVFilter::PVFieldsSplitterParamWidget>();
		QString registered_name = in_t_cpy->registered_name();
        PVLOG_DEBUG(" type_name %s, %s\n", qPrintable(in_t_cpy->type_name()), qPrintable(registered_name));
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
    QFile f(urlFile);
    if (f.exists()) {//if the file exists...
        myTreeModel->openXml(urlFile); //open it
    }
}




/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::slotSave
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::slotSave() {
    QModelIndex index;
    myTreeView->applyModification(myParamBord_old_model,index);
    QFileDialog fd;
     //open file chooser
    QString urlFile = fd.getSaveFileName(0,QString("Select the file."),PVRush::normalize_get_helpers_plugins_dirs(QString("text")).first());
	if (!urlFile.isEmpty()) {
		myTreeModel->saveXml(urlFile); //save file
	}
}



/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::slotUpdateToolDesabled
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::slotUpdateToolDesabled(const QModelIndex &index){
    PVRush::PVXmlTreeNodeDom *node = myTreeModel->nodeFromIndex(index);
    
    hideParamBoard();
    
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
        file->addSeparator();
		PVInputTypeMenuEntries::add_inputs_to_menu(file, this, SLOT(slotOpenLog()));
        file->addSeparator();


        QMenu *splitter = menuBar->addMenu(tr("&Splitter"));
        splitter->addAction(actionAddUrl);
        splitter->addAction(actionAddRegExAfter);    
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
		QMessageBox box(QMessageBox::Critical, tr("Error"), tr("No input plugins can manage the source file '%1'. Aborting...").arg(in_t->human_name_of_input(_log_input)));
		_log_input = PVCore::PVArgument(); // No log input
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

void PVInspector::PVXmlEditorWidget::create_extractor()
{
	if (_log_extract) {
		_log_extract->force_stop_controller();
	}
	_log_extract.reset(new PVRush::PVExtractor());
	_log_extract->start_controller();
	_log_source = _log_sc->create_source_from_input(_log_input);
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

	// TODO: QMessageBox ask;
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
        PVLOG_DEBUG("PVFilter::PVFieldSplitterCSVParamWidget::hideParamBoard()\n");
}

void PVInspector::PVXmlEditorWidget::set_format_from_dom()
{
	QDomElement const& rootDom = myTreeModel->getRootDom();
	PVRush::PVFormat format;
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
	assert(end > start);
	_nraw_model->set_consistent(false);
	set_format_from_dom();
	// Create the nraw thanks to the extractor
	PVRush::PVControllerJob_p job = _log_extract->process_from_agg_idxes(start, end);
	job->wait_end();
	_log_extract->dump_nraw();
	_nraw_model->set_nraw(_log_extract->get_nraw());
	_nraw_model->set_consistent(true);
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
