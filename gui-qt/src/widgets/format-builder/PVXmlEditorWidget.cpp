//! \file PVXmlEditorWidget.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011


#include <PVXmlEditorWidget.h>
#include <PVXmlTreeItemDelegate.h>
#include <pvcore/PVXmlTreeNodeDom.h>
#include <PVXmlParamWidget.h>




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
    myParamBord = new PVXmlParamWidget();

 
    vbParam->addWidget(myParamBord);  

    
    
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
    actionOpenLog = new QAction(tr("Open a log"),(QObject*)this);
    actionOpenLog->setIcon(QIcon(":/log-icon"));
}


/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::initConnexions
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::initConnexions() {
    //connexion to update the parameter board
    connect(myTreeView, SIGNAL(clicked(const QModelIndex &)), myParamBord, SLOT(edit(const QModelIndex &)));
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
    connect(actionOpenLog,SIGNAL(triggered()),this,SLOT(slotOpenLog()));
    connect(actionSave, SIGNAL(triggered()), this, SLOT(slotSave()));
    connect(actionAddUrl, SIGNAL(triggered()), this, SLOT(slotAddUrl()));
    connect(myParamBord,SIGNAL(signalNeedApply()),this,SLOT(slotNeedApply()));
    connect(myParamBord,SIGNAL(signalSelectNext()),myTreeView,SLOT(slotSelectNext()));
    
    

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
    tools->addAction(actionOpenLog);
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
                PVLOG_INFO("name: %s\n", qPrintable(it.key()));
                PVFilter::PVFieldsSplitterParamWidget_p pluginsSplitter = it.value()/*->clone<PVFilter::PVFieldsSplitterParamWidget>()*/;
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
        PVFilter::PVFieldsSplitterParamWidget_p in_t_cpy = in_t.get()->clone<PVFilter::PVFieldsSplitterParamWidget>();
        
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
    myTreeView->applyModification(myParamBord,index);
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
        myParamBord->drawForNo(ind);
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
    myTreeView->applyModification(myParamBord,index);
}


/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::slotOpen
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::slotOpen(){
    QFileDialog fd;
    //open file chooser
    QString urlFile = fd.getOpenFileName(0,QString("Select the file."),PVRush::normalize_get_helpers_plugins_dirs(QString("text")).first());
    QFile f(urlFile);
    if(f.exists()){//if the file exists...
      myTreeModel->openXml(urlFile);//open it
    }
}


/******************************************************************************
 *
 * PVInspector::PVXmlEditorWidget::slotSave
 *
 *****************************************************************************/
void PVInspector::PVXmlEditorWidget::slotSave() {
    QModelIndex index;
    myTreeView->applyModification(myParamBord,index);
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
    PVCore::PVXmlTreeNodeDom *node = myTreeModel->nodeFromIndex(index);
    
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
        file->addSeparator();


        file->addAction(actionOpenLog);
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

