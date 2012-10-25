/**
 * \file PVFormatBuilderWidget.cpp
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#include <QSplitter>

#include <PVFormatBuilderWidget.h>
#include <PVXmlTreeItemDelegate.h>
#include <PVXmlParamWidget.h>
#include <pvguiqt/PVInputTypeMenuEntries.h>

#include <pvkernel/rush/PVNormalizer.h>
#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <pvkernel/rush/PVSourceCreatorFactory.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/filter/PVFieldSplitterChunkMatch.h>

#include <pvguiqt/PVAxesCombinationWidget.h>
#include <pvkernel/core/PVRecentItemsManager.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVFuncObserver.h>
#include <pvhive/PVCallHelper.h>

#define FORMAT_BUILDER_TITLE (QObject::tr("Format builder"))
/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::PVFormatBuilderWidget
 *
 *****************************************************************************/
PVInspector::PVFormatBuilderWidget::PVFormatBuilderWidget(QWidget * parent):
	QMainWindow(parent)
{
	init(parent);
	setObjectName("PVFormatBuilderWidget");
}

bool PVInspector::PVFormatBuilderWidget::somethingChanged(void)
{
	if (myTreeView->model()->rowCount()) {
		return true;
	}

	return false;
}

void PVInspector::PVFormatBuilderWidget::closeEvent(QCloseEvent *event)
{
	if (somethingChanged()) {
		QMessageBox msgBox;
		msgBox.setText("The document has been modified.");
		msgBox.setInformativeText("Do you want to save your changes?");
		msgBox.setStandardButtons(QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
		msgBox.setDefaultButton(QMessageBox::Save);
		int ret = msgBox.exec();
		switch (ret) {
		case QMessageBox::Save:
			if (save()) {
	       			event->accept();
			} else {
				event->ignore();
			}
			break;
		case QMessageBox::Discard:
			event->accept();
			break;
		case QMessageBox::Cancel:
			event->ignore();
			break;
		default:
			// should never be reached
			break;
		}

	} else {
		event->accept();
	}
}

void PVInspector::PVFormatBuilderWidget::init(QWidget* parent)
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
    
    //initialisation of the splitters list
    initSplitters();

    menuBar =new QMenuBar();
    initMenuBar();
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
	_main_tab = new QTabWidget();
	QWidget* vb_widget = new QWidget();
	vb_widget->setLayout(vb);
	_main_tab->addTab(vb_widget, tr("Filters"));

	_axes_comb_widget = new PVGuiQt::PVAxesCombinationWidget(myTreeModel->get_axes_combination());
	_main_tab->addTab(_axes_comb_widget, tr("Axes combination"));

	main_splitter->addWidget(_main_tab);

	// Tab widget for the NRAW
	QTabWidget* nraw_tab = new QTabWidget();
	nraw_tab->addTab(_nraw_widget, tr("Normalization preview"));
	main_splitter->addWidget(nraw_tab);

	_inv_lines_widget = new QListWidget();
	nraw_tab->addTab(_inv_lines_widget, tr("Unmatched lines"));

	QVBoxLayout* main_layout = new QVBoxLayout();
	initToolBar(main_layout);
	main_layout->addWidget(main_splitter);
	main_layout->setMenuBar(menuBar);

	QWidget *central_widget = new QWidget();
	central_widget->setLayout(main_layout);

	setCentralWidget(central_widget);
    
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
	// QRect geom = QRect(0,0,700,500);
	// setWindowFlags(Qt::Window);
	// geom.moveCenter(parent->geometry().center());
	// setGeometry(geom);
}
/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::~PVFormatBuilderWidget
 *
 *****************************************************************************/
PVInspector::PVFormatBuilderWidget::~PVFormatBuilderWidget() {
    /*actionAddFilterAfter->deleteLater();
    actionAddRegExAfter->deleteLater();
    actionDelete->deleteLater();
    myTreeView->deleteLater();*/
}


/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::actionAllocation
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::actionAllocation(){
    actionAddAxisIn = new QAction("add an axis",(QObject*)this);
    actionAddAxisIn->setIcon(QIcon(":/add-axis"));
    actionAddFilterAfter = new QAction("add a filter",(QObject*)this);
    actionAddFilterAfter->setIcon(QIcon(":/filter"));
    actionAddRegExAfter = new QAction("add a RegEx",(QObject*)this);
    actionAddRegExAfter->setIcon(QIcon(":/add-regexp"));
    actionAddUrl = new QAction("add an URL",(QObject*)this);
    actionAddUrl->setIcon(QIcon(":/add-url"));

    actionSave = new QAction("&Save format",(QObject*)this);
    actionSave->setShortcut(QKeySequence(Qt::CTRL+Qt::Key_S));
    actionSave->setIcon(QIcon(":/save"));
    actionSaveAs = new QAction("Save format as...",(QObject*)this);
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
    actionOpen = new QAction(tr("Open format..."),(QObject*)this);
    actionOpen->setShortcut(QKeySequence(Qt::CTRL+Qt::Key_O));
    actionOpen->setIcon(QIcon(":/document-open.png"));

    actionNewWindow = new QAction(tr("New window"),(QObject*)this);
    actionCloseWindow = new QAction(tr("Close window"),(QObject*)this);
    actionCloseWindow->setShortcut(QKeySequence(Qt::CTRL+Qt::Key_W));

}


/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::initConnexions
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::initConnexions() {
    //connexion to update the parameter board
    connect(myTreeView, SIGNAL(clicked(const QModelIndex &)), myParamBord_old_model, SLOT(edit(const QModelIndex &)));
    //connexion to endable/desable items in toolsbar menu.
    connect(myTreeView, SIGNAL(clicked(const QModelIndex &)), this, SLOT(slotUpdateToolDesabled(const QModelIndex &)));
    
    //data has changed from tree 
    connect(myTreeModel, SIGNAL(dataChanged(const QModelIndex &, const QModelIndex& )), myTreeView, SLOT(slotDataHasChanged(const QModelIndex & , const QModelIndex & )));
    
	// When an item is clicked in the tree view, auto-select the good axis in the mini-extractor
	connect(myTreeView, SIGNAL(clicked(const QModelIndex &)), this, SLOT(slotItemClickedInView(const QModelIndex &)));
	

    /*
     * Connexions for both menu and toolbar.
     */
    connect(actionAddAxisIn,  SIGNAL(triggered()),this,SLOT(slotAddAxisIn()));
    connect(actionAddFilterAfter, SIGNAL(triggered()), this, SLOT(slotAddFilterAfter()));
    connect(actionAddRegExAfter, SIGNAL(triggered()), this, SLOT(slotAddRegExAfter()));
    connect(actionDelete, SIGNAL(triggered()), this, SLOT(slotDelete()));
    connect(actionMoveDown,SIGNAL(triggered()),this,SLOT(slotMoveDown()));
    connect(actionMoveUp,SIGNAL(triggered()),this,SLOT(slotMoveUp()));
    connect(actionNewWindow,SIGNAL(triggered()),this,SLOT(slotNewWindow()));
    connect(actionCloseWindow,SIGNAL(triggered()),this,SLOT(close()));
    connect(actionOpen,SIGNAL(triggered()),this,SLOT(slotOpen()));
    connect(actionSave, SIGNAL(triggered()), this, SLOT(slotSave()));
    connect(actionSaveAs, SIGNAL(triggered()), this, SLOT(slotSaveAs()));
    connect(actionAddUrl, SIGNAL(triggered()), this, SLOT(slotAddUrl()));
    connect(myParamBord_old_model,SIGNAL(signalNeedApply()),this,SLOT(slotNeedApply()));
    connect(myParamBord_old_model,SIGNAL(signalSelectNext()),myTreeView,SLOT(slotSelectNext()));
    
    // Connections for the axes combination editor
	connect(_main_tab, SIGNAL(currentChanged(int)), this, SLOT(slotMainTabChanged(int)));

}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::initToolBar
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::initToolBar(QVBoxLayout *vb){

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
 * PVInspector::PVFormatBuilderWidget::initSplitters
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::initSplitters() {
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
 * PVInspector::PVFormatBuilderWidget::slotAddAxisIn
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotAddAxisIn() {
    myTreeView->addAxisIn();
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotAddFilterAfter
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotAddFilterAfter() {
    myTreeView->addFilterAfter();
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotAddRegExAfter
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotAddRegExAfter() {
    myTreeView->addRegExIn();
}
/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotAddSplitter
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotAddSplitter()
{
        QAction* action_src = (QAction*) sender();
        QString const& itype = action_src->data().toString();
        PVFilter::PVFieldsSplitterParamWidget_p in_t = LIB_CLASS(PVFilter::PVFieldsSplitterParamWidget)::get().get_class_by_name(itype);
        PVFilter::PVFieldsSplitterParamWidget_p in_t_cpy = in_t->clone<PVFilter::PVFieldsSplitterParamWidget>();
		QString registered_name = in_t_cpy->registered_name();
        PVLOG_DEBUG("(PVInspector::PVFormatBuilderWidget::slotAddSplitter) type_name %s, %s\n", qPrintable(in_t_cpy->type_name()), qPrintable(registered_name));
        myTreeView->addSplitter(in_t_cpy);
}


/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotAddUrl
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotAddUrl(){
    myTreeView->addUrlIn();
}


/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotApplyModification
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotApplyModification() {
  QModelIndex index;
    myTreeView->applyModification(myParamBord_old_model,index);
}


/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotDelete
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotDelete() {
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
 * PVInspector::PVFormatBuilderWidget::slotMoveUp
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotMoveUp() {
    myTreeView->moveUp();
}


/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotMoveDown
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotMoveDown() {
    myTreeView->moveDown();
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotNeedApply
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotNeedApply(){
  QModelIndex index;
    myTreeView->applyModification(myParamBord_old_model,index);
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotNewWindow
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotNewWindow() 
{
	PVFormatBuilderWidget *other = new PVFormatBuilderWidget;
	other->move(x() + 40, y() + 40);
	other->show();
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotOpen
 *
 *****************************************************************************/
QString PVInspector::PVFormatBuilderWidget::slotOpen() {
    //open file chooser
    QString urlFile = _open_dialog.getOpenFileName(0, QString("Select the file."), PVRush::normalize_get_helpers_plugins_dirs(QString("text")).first());
	bool valid = openFormat(urlFile);

	return valid ? urlFile : QString();
}


bool PVInspector::PVFormatBuilderWidget::save() {
	// Take focus, so any currently edited argument will be set
	setFocus(Qt::MouseFocusReason);

	if (_cur_file.isEmpty()) {
		return saveAs();
	}	

	bool save_xml = myTreeModel->saveXml(_cur_file);
	if (save_xml) {
		PVHive::call<FUNC(PVCore::PVRecentItemsManager::add)>(PVCore::PVRecentItemsManager::get(), _cur_file, PVCore::PVRecentItemsManager::Category::EDITED_FORMATS);
		return true;
	}

	QMessageBox err(QMessageBox::Question, tr("Error while saving format"), tr("Unable to save the changes to %1. Do you want to save this format to another location ?").arg(_cur_file), QMessageBox::Yes | QMessageBox::No);
	if (err.exec() == QMessageBox::No) {
		return false;
	}

	return saveAs();
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotSave
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotSave() {
	save();
}

bool PVInspector::PVFormatBuilderWidget::saveAs() {
	setFocus(Qt::MouseFocusReason);

    QModelIndex index;
    myTreeView->applyModification(myParamBord_old_model,index);
     //open file chooser
    QString urlFile = _save_dialog.getSaveFileName(0,QString("Select the file."),PVRush::normalize_get_helpers_plugins_dirs(QString("text")).first());
	if (!urlFile.isEmpty()) {
		if (myTreeModel->saveXml(urlFile)) {
			_cur_file = urlFile;
			setWindowTitleForFile(urlFile);
			PVHive::call<FUNC(PVCore::PVRecentItemsManager::add)>(PVCore::PVRecentItemsManager::get(), urlFile, PVCore::PVRecentItemsManager::Category::EDITED_FORMATS);
			return true;
		}
	}

	return false;
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotSaveAs
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotSaveAs() {
	saveAs();
}

void PVInspector::PVFormatBuilderWidget::setWindowTitleForFile(QString const& path)
{
	// Change the window title with the filename of the format
	QFileInfo fi(path);
	setWindowTitle(FORMAT_BUILDER_TITLE + QString(" - ") + fi.fileName());
}


/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotUpdateToolDesabled
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotUpdateToolDesabled(const QModelIndex &index){
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
 * PVInspector::PVFormatBuilderWidget::initMenuBar
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::initMenuBar() {
        QMenu *file = menuBar->addMenu(tr("&File"));

	file->addAction(actionNewWindow);
        file->addSeparator();
        file->addAction(actionOpen);
        file->addAction(actionSave);
        file->addAction(actionSaveAs);
        file->addSeparator();
		PVGuiQt::PVInputTypeMenuEntries::add_inputs_to_menu(file, this, SLOT(slotOpenLog()));
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

        file->addSeparator();
	file->addAction(actionCloseWindow);

}


/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotOpenLog
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotOpenLog()
{
	PVRush::PVInputType_p in_t = PVGuiQt::PVInputTypeMenuEntries::input_type_from_action((QAction*) sender());
	PVRush::list_creators lcr = PVRush::PVSourceCreatorFactory::get_by_input_type(in_t);

	QString choosenFormat;
	PVRush::PVInputType::list_inputs inputs;
	PVRush::hash_formats formats, new_formats;

	PVCore::PVArgumentList args;
	if (!in_t->createWidget(formats, new_formats, inputs, choosenFormat, args, this))
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
		create_extractor();
		for (itcr = lcr.begin(); itcr != lcr.end(); itcr++) {
			PVRush::PVSourceCreator_p sc = *itcr;
			if (sc->pre_discovery(_log_input)) {
				try {
					_log_sc = sc;
					// The moni-extractor use the discovery source, as not that much processing is done (it can be handle locally for instance !)
					_log_source = _log_sc->create_discovery_source_from_input(_log_input, PVRush::PVFormat());
				}
				catch (PVRush::PVFormatInvalid& e) {
					_log_sc.reset();
					continue;
				}
				break;
			}
		}

		if (!_log_sc) {
			_log_input = PVRush::PVInputDescription_p();
			QMessageBox box(QMessageBox::Critical, tr("Error"), tr("No input plugins can manage the source file '%1'. Aborting...").arg(in_t->human_name_of_input(_log_input)));
			box.show();
			return;
		}
		_log_extract->add_source(_log_source);

		_log_input_type = in_t;

		// First extraction
		if (is_dom_empty()) {
			guess_first_splitter();
		}

		update_table(FORMATBUILDER_EXTRACT_START_DEFAULT, FORMATBUILDER_EXTRACT_END_DEFAULT);

	}
	catch (PVRush::PVInputException &e) {
		_log_input = PVRush::PVInputDescription_p();
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

void PVInspector::PVFormatBuilderWidget::create_extractor()
{
	if (_log_extract) {
		_log_extract->force_stop_controller();
	}
	_log_extract.reset(new PVRush::PVExtractor());
	_log_extract->dump_all_elts(true);
	_log_extract->dump_inv_elts(true);
	_log_extract->start_controller();
}

void PVInspector::PVFormatBuilderWidget::guess_first_splitter()
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
 * PVInspector::PVFormatBuilderWidget::hideParamBoard
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::hideParamBoard(){
        PVLOG_DEBUG("PVInspector::PVFormatBuilderWidget::hideParamBoard\n");
}

void PVInspector::PVFormatBuilderWidget::set_format_from_dom()
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
 * PVInspector::PVFormatBuilderWidget::showParamBoard
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::showParamBoard(PVRush::PVXmlTreeNodeDom *node){
        assert(node);
        PVLOG_DEBUG("PVInspector::PVFormatBuilderWidget::showParamBoard()\n");
        
        //myParamBord=node->getSplitterPlugin()->get_param_widget();

}

void PVInspector::PVFormatBuilderWidget::update_table(PVRow start, PVRow end)
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
	QStringList const& elts_invalid = job->get_invalid_elts();
	QStringList::const_iterator it_ie;
	for (it_ie = elts_invalid.begin(); it_ie != elts_invalid.end(); it_ie++) {
		QString const& line = *it_ie;
		_inv_lines_widget->addItem(line);
	}
}

void PVInspector::PVFormatBuilderWidget::slotExtractorPreview()
{
	PVRow start,end;
	_nraw_widget->get_ext_args(start,end);
	update_table(start,end);
}

bool PVInspector::PVFormatBuilderWidget::is_dom_empty()
{
	QDomElement const& rootDom = myTreeModel->getRootDom();
	return !rootDom.hasChildNodes();
}

void PVInspector::PVFormatBuilderWidget::slotItemClickedInView(const QModelIndex &index)
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

void PVInspector::PVFormatBuilderWidget::set_axes_name_selected_row_Slot(int row)
{
	PVRush::PVNraw const& nraw = _log_extract->get_nraw();
	// We could use QList::fromVector(QVector::fromStdVector(nraw_table_line)), but that's not really efficient...
	if (row >= nraw.get_number_rows()) {
		PVLOG_WARN("(PVFormatBuilderWidget::set_axes_name_selected_row_Slot) row index '%d' does not exist in the current NRAW (size '%d').\n", row, nraw.get_number_rows());
		return;
	}
	/*
	QStringList names;
	PVRush::PVNraw::const_nraw_table_line line = nraw.get_table().get_row(row);
	for (PVCol j = 0; j < line.size(); j++) {
		// We need to do a deep copy of this
		QString const& v = line[j].get_qstr();
		QString deep_copy((const QChar*) v.constData(), v.size());
		names << deep_copy;
	}
	myTreeModel->setAxesNames(names);*/
}

void PVInspector::PVFormatBuilderWidget::set_axes_type_selected_row_Slot(int row)
{
}

bool PVInspector::PVFormatBuilderWidget::openFormat(QString const& path)
{
    QFile f(path);
    if (f.exists()) {//if the file exists...
        if (myTreeModel->openXml(path)) {
			_cur_file = path;
			setWindowTitleForFile(path);
			return true;
		}
    }

    return false;
}

void PVInspector::PVFormatBuilderWidget::openFormat(QDomDocument& doc)
{
	myTreeModel->openXml(doc);
}

void PVInspector::PVFormatBuilderWidget::slotMainTabChanged(int idx)
{
	if (idx == 1) {
		// This is the axes combination editor.

		// Get the list of axes and update the axis combination
		myTreeModel->updateAxesCombination();
		//_axes_comb_widget->update_all();
	}
}
