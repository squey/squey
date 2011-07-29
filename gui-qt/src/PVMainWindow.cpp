//! \file PVMainWindow.cpp
//! $Id: PVMainWindow.cpp 3251 2011-07-06 11:51:57Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011


#include <QtCore>
#include <QtGui>

#include <QVBoxLayout>
#include <QDialog>
#include <QMessageBox>
#include <QFrame>
#include <QFuture>
#include <QFutureWatcher>
#include <QVBoxLayout>
#include <QMenuBar>
#include <QLine>
#include <QDialogButtonBox>

#include <PVMainWindow.h>
#include <PVExtractorWidget.h>
#include <PVFilterSearchWidget.h>
#include <PVFilesTypesSelWidget.h>
#include <PVStringListChooserWidget.h>
#include <PVArgumentListWidget.h>
#include <PVInputTypeMenuEntries.h>
//#include <geo/GKMapView.h>

#ifdef CUSTOMER_RELEASE
  #ifdef WIN32
    #include <winlicensesdk.h>
  #endif
#endif	// CUSTOMER_RELEASE

#include <pvcore/general.h>
#include <pvcore/debug.h>
#include <pvcore/PVAxisIndexType.h>
#include <pvcore/PVClassLibrary.h>
#include <pvcore/PVMeanValue.h>

#include <pvrush/PVInput.h>
#include <pvrush/PVNormalizer.h>
#include <pvrush/PVSourceCreator.h>
#include <pvrush/PVSourceCreatorFactory.h>


#include <picviz/general.h>
#include <picviz/arguments.h>
#include <picviz/PVSelection.h>
#include <picviz/PVMapping.h>
#include <picviz/PVPlotting.h>
#include <picviz/PVStateMachine.h>


#include <PVProgressBox.h>

// Filters
#include <pvfilter/PVFilterLibrary.h>

#include <pvgl/general.h>
#include <pvgl/PVCom.h>
#include <pvgl/PVMain.h>

#include <PVXmlEditorWidget.h>

/******************************************************************************
 *
 * PVInspector::PVMainWindow::PVMainWindow
 *
 *****************************************************************************/
PVInspector::PVMainWindow::PVMainWindow(QWidget *parent) : QMainWindow(parent)
{
	QSplashScreen splash(QPixmap(":/splash-screen"));

	splash.show();

	PVLOG_DEBUG("%s: Creating object\n", __FUNCTION__);

	about_dialog = 0;
	// picviz_datatreerootitem_t *datatree;

	setGeometry(20,10,800,600);
//	datatree = picviz_datatreerootitem_new();

	/* This does not exist yet :-) */
	current_tab = NULL;

	//import_source = NULL;

	//We activate all available Windows
	pv_AxisProperties = new PVAxisPropertiesWidget(this);
	pv_AxisProperties->hide();

	pv_ColorDialog = new PVColorDialog(this);
	pv_ColorDialog->setOption(QColorDialog::ShowAlphaChannel, true);
	pv_ColorDialog->hide();

	pv_ExportSelectionDialog = new PVExportSelectionDialog(this);
	pv_ExportSelectionDialog->hide();
	root = Picviz::PVRoot_p(new Picviz::PVRoot());
	pv_FilterWidget = new PVFilterWidget(this);
	pv_FilterWidget->hide();

	pv_ImportFileDialog = new PVImportFileDialog(this);
	pv_ImportFileDialog->hide();

	// pv_FilterSearchWidget = new PVInspector::PVFilterSearchWidget(this);
	// pv_FilterSearchWidget->hide();
				// pv_RemoteLog = new LogViewerWidget(this);
	// pv_RemoteLog->resize(500,60);
	// // pv_RemoteLog->hide();

//	pv_MapWidget = new PVMapWidget(this);
	//pv_MapWidget->hide();

	pv_OpenFileDialog = new PVOpenFileDialog(this);
	pv_OpenFileDialog->hide();

	pv_SaveFileDialog = new PVSaveFileDialog(this);
	pv_SaveFileDialog->hide();


	pv_ListingsTabWidget = new PVListingsTabWidget(this, this);


	// We display the PV Icon together with a button to import files
	pv_centralWidget = new QWidget(this);

	pv_mainLayout = new QVBoxLayout();
	pv_mainLayout->setAlignment(Qt::AlignCenter);
	pv_mainLayout->setSpacing(40);
	pv_mainLayout->setContentsMargins(0,0,0,0);

	pv_welcomeIcon = new QPixmap(":/logo.png");
	pv_labelWelcomeIcon = new QLabel(this);
	pv_labelWelcomeIcon->setPixmap(*pv_welcomeIcon);
	pv_labelWelcomeIcon->resize(pv_welcomeIcon->width(), pv_welcomeIcon->height());

	pv_ImportFileButton = new QPushButton("Import files...");
	pv_ImportFileButton->setIcon(QIcon(":/document-new.png"));

	connect(pv_ImportFileButton, SIGNAL(clicked()), this, SLOT(import_type_default_Slot()));
	connect(pv_ListingsTabWidget, SIGNAL(is_empty()), this, SLOT(display_icon_Slot()) );

	pv_mainLayout->addWidget(pv_labelWelcomeIcon);
	pv_mainLayout->addWidget(pv_ImportFileButton);
	pv_mainLayout->addWidget(pv_ListingsTabWidget);
	
	pv_ListingsTabWidget->hide();
	pv_centralWidget->setLayout(pv_mainLayout);
	setCentralWidget(pv_centralWidget);

	pv_ListingsTabWidget->setFocus(Qt::OtherFocusReason);


	// RemoteLogDialog = new QMainWindow(this, Qt::Dialog);
	// QObject::connect(RemoteLogDialog, SIGNAL(destroyed()), this, SLOT(hide()));

	//We populate all actions, menus and connect them
	create_actions();
	create_menus();
	connect_actions();
	connect_widgets();
	menu_activate_is_file_opened(false);
	
	update_check();

	create_pvgl_thread ();

	statusBar();
	splash.finish(pv_ImportFileButton);


	// Center the main window
	QRect r = geometry();
	r.moveCenter(QApplication::desktop()->screenGeometry(this).center());
	setGeometry(r);
}

void PVInspector::PVMainWindow::closeEvent(QCloseEvent* event)
{
	pvgl_thread->terminate();
	pvgl_thread->wait();
	delete pvgl_thread;
	// Gracefull stops PVGL::PVMain
	PVGL::PVMain::stop();
	QMainWindow::closeEvent(event);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::check_messages
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::check_messages()
{ // FIXME: here we use the current tab. We should really use a tab calculated for the pv_view of the message.
	
	PVGL::PVMessage message;
	if (pvgl_com->get_message_for_qt(message)) {
		//PVLOG_INFO("PVInspector::PVMainWindow::check_messages()\n");
		switch (message.function) {
			case PVGL_COM_FUNCTION_CLEAR_SELECTION:
				{
					/* FIXME !!!! We've killed the Listing window! pv_ListingWindow->pv_listing_view->clearSelection();*/
					if (!current_tab)
						break;
					//PVLOG_INFO("PVInspector::PVMainWindow::check_messages : PVGL_COM_FUNCTION_CLEAR_SELECTION\n");
					current_tab->refresh_listing_Slot();
					current_tab->repaint(0,0,-1,-1);
					break;
				}
			case PVGL_COM_FUNCTION_REFRESH_LISTING:
				{
					//PVLOG_INFO("PVInspector::PVMainWindow::check_messages : PVGL_COM_FUNCTION_REFRESH_LISTING\n");
					message.pv_view->process_visibility();
					if (!current_tab)
						break;
					current_tab->refresh_listing_with_horizontal_header_Slot();
					current_tab->update_pv_listing_model_Slot();
					current_tab->refresh_listing_Slot();
					break;
				}
			case PVGL_COM_FUNCTION_SELECTION_CHANGED:
				{
					// FIXME DDX! update_row_count_in_all_dynamic_listing_model_Slot();
					//PVLOG_INFO("PVInspector::PVMainWindow::check_messages : PVGL_COM_FUNCTION_SELECTION_CHANGED\n");
					if (!current_tab)
						break;
					current_tab->selection_changed_Slot();
					current_tab->refresh_listing_with_horizontal_header_Slot();
					current_tab->update_pv_listing_model_Slot();
					current_tab->refresh_listing_Slot();
					break;
				}
			case PVGL_COM_FUNCTION_SCREENSHOT_CHOOSE_FILENAME:
						{
							QString initial_path = QDir::currentPath();

							QString screenshot_filename;
							screenshot_filename = pv_ListingsTabWidget->tabText(pv_ListingsTabWidget->currentIndex());
							screenshot_filename.append("_%1.png");
							screenshot_filename = screenshot_filename.arg(current_tab->get_screenshot_index(), 3, 10, QString("0")[0]);
							current_tab->increment_screenshot_index();
							initial_path += "/" + screenshot_filename;

							QString *filename = new QString (QFileDialog::getSaveFileName(this, tr("Save Screenshot As"), initial_path, tr("PNG Files (*.png);;All Files (*)")));
							if (!filename->isEmpty()) {
								message.function = PVGL_COM_FUNCTION_TAKE_SCREENSHOT;
								message.pointer_1 = filename;
								pvgl_com->post_message_to_gl(message);
							}
						}
					break;
			case PVGL_COM_FUNCTION_SCREENSHOT_TAKEN:
						{
							QString *filename = reinterpret_cast<QString *>(message.pointer_1);
							PVLOG_INFO("%s: destroying the screenshot filename: %s\n", __FUNCTION__, filename->toStdString().c_str());
							delete filename;
						}
					break;
			case PVGL_COM_FUNCTION_ONE_VIEW_DESTROYED:
						{
							QString *name = reinterpret_cast<QString *>(message.pointer_1);
							PVLOG_DEBUG("%s: Should remove the window menu entry named: >%s<\n", __FUNCTION__, qPrintable(*name));
							QList<QAction *> all_actions = windows_Menu->actions ();
							for (QList<QAction*>::iterator it = all_actions.begin(); it != all_actions.end(); ++it) {
								QAction *action = *it;
								if (action->text() == *name) {
									windows_Menu->removeAction(action);
								}
							}
							delete name;
						}
					break;
			case PVGL_COM_FUNCTION_VIEWS_DESTROYED:
					// Check that everyone has released its objects, and that the smart pointer will be deleted !!
					if (message.pv_view.use_count() != 1) {
						PVLOG_WARN("PVGL_COM_FUNCTION_VIEWS_DESTROYED: in PVMainWindow, after views destroyed, PVView has a use count of %d (should be 1)\n", message.pv_view.use_count());
					}
					if (message.pv_view->get_mapped_parent().use_count() != 2) {
						PVLOG_WARN("PVGL_COM_FUNCTION_VIEWS_DESTROYED: in PVMainWindow, after views destroyed, PVMapped has a use count of %d (should be 2)\n", message.pv_view->get_mapped_parent().use_count());
					}
					if (message.pv_view->get_plotted_parent().use_count() != 2) {
						PVLOG_WARN("PVGL_COM_FUNCTION_VIEWS_DESTROYED: in PVMainWindow, after views destroyed, PVPlotted has a use count of %d (should be 2)\n", message.pv_view->get_plotted_parent().use_count());
					}
					if (message.pv_view->get_plotted_parent()->get_source_parent().use_count() != 2) {
						PVLOG_WARN("PVGL_COM_FUNCTION_VIEWS_DESTROYED: in PVMainWindow, after views destroyed, PVSource has a use count of %d (should be 2)\n", message.pv_view->get_plotted_parent()->get_source_parent().use_count());
					}
					break;
			case PVGL_COM_FUNCTION_VIEW_CREATED:
						{
							QString *name = reinterpret_cast<QString *>(message.pointer_1);

							windows_Menu->addAction(new QAction(*name, this));
							PVLOG_DEBUG("%s: destroying the view name (after pvgl view creation) : %s\n", __FUNCTION__, qPrintable(*name));
							delete name;
						}
					break;
			case PVGL_COM_FUNCTION_COMMIT_SELECTION_IN_CURRENT_LAYER:
					commit_selection_in_current_layer(message.pv_view);
					break;
			case PVGL_COM_FUNCTION_COMMIT_SELECTION_IN_NEW_LAYER:
					commit_selection_to_new_layer(message.pv_view);
					break;
			case PVGL_COM_FUNCTION_SET_COLOR:
					set_color(message.pv_view);
					break;
			default:
					PVLOG_ERROR("%s: Unknow function in message: %d\n", __FUNCTION__, message.function);
		}
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::commit_selection_in_current_layer
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::commit_selection_in_current_layer(Picviz::PVView_p picviz_view)
{
	//Picviz::StateMachine *state_machine = NULL;

	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);

	//state_machine = picviz_view->state_machine;

	/* We get the current selected layer */
	Picviz::PVLayer &current_selected_layer = picviz_view->layer_stack.get_selected_layer();
	/* We fill it's lines_properties */
	picviz_view->output_layer.get_lines_properties().A2B_copy_restricted_by_selection_and_nelts(current_selected_layer.get_lines_properties(), picviz_view->real_output_selection, picviz_view->row_count);
	/* We need to process the view from the layer_stack */
	picviz_view->process_from_layer_stack();
	/* We refresh the PVView_p */
	update_pvglview(picviz_view, PVGL_COM_REFRESH_SELECTION);
	/* We refresh the listing */
	//FIXME! This should be done in a function
	for (int i = 0; i < pv_ListingsTabWidget->count();i++) {
		PVTabSplitter *tab = dynamic_cast<PVTabSplitter*>(pv_ListingsTabWidget->widget(i));
		if (!tab) {
			PVLOG_ERROR("PVInspector::PVMainWindow::%s: Tab isn't tab!!!\n", __FUNCTION__);
		} else {
			if (tab->get_lib_view() == picviz_view) {
				tab->refresh_listing_Slot();
			}
		}
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::commit_selection_to_new_layer
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::commit_selection_to_new_layer(Picviz::PVView_p picviz_view)
{
	/* We also need an access to the state machine */
	//Picviz::StateMachine *state_machine = NULL;

	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);

	//state_machine = picviz_view->state_machine;

	/* We FIRST do what has to be done in the Lib */
	/* We create a new layer */
	picviz_view->layer_stack.append_new_layer();
	Picviz::PVLayer &new_layer = picviz_view->layer_stack.get_selected_layer();
	/* We set it's selection to the final selection */
	picviz_view->set_selection_with_final_selection(new_layer.get_selection());
	picviz_view->output_layer.get_lines_properties().A2B_copy_restricted_by_selection_and_nelts(new_layer.get_lines_properties(), new_layer.get_selection(), picviz_view->row_count);
	// picviz_lines_properties_A2B_copy_restricted_by_selection_and_nelts(picviz_view->output_layer.get_lines_properties(), new_layer->lines_properties, new_layer.get_selection(), picviz_view->row_count);
	/* THEN we can do the updates */
	/* We need to reprocess the layer stack */
	picviz_view->process_from_layer_stack();

	refresh_view(picviz_view);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::connect_actions()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::connect_actions()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);
	connect(export_file_Action, SIGNAL(triggered()), this, SLOT(export_file_Slot()));
	connect(export_selection_Action, SIGNAL(triggered()), this, SLOT(export_selection_Slot()));
	//connect(import_file_Action, SIGNAL(triggered()), this, SLOT(import_file_Slot()));
	connect(extractor_file_Action, SIGNAL(triggered()), this, SLOT(extractor_file_Slot()));
	connect(new_file_Action, SIGNAL(triggered()), this, SLOT(new_file_Slot()));
	connect(new_scene_Action, SIGNAL(triggered()), this, SLOT(new_scene_Slot()));
//	connect(remote_log_Action, SIGNAL(triggered()), this, SLOT(remote_log_Slot()));

// #ifdef CUSTOMER_RELEASE
// 	connect(open_file_Action, SIGNAL(triggered()), this, SLOT(open_file_Slot()));
// 	connect(save_file_Action, SIGNAL(triggered()), this, SLOT(save_file_Slot()));
// #endif
	connect(quit_Action, SIGNAL(triggered()), this, SLOT(quit_Slot()));
	connect(select_scene_Action, SIGNAL(triggered()), this, SLOT(select_scene_Slot()));

	connect(view_open_Action, SIGNAL(triggered()), this, SLOT(view_open_Slot()));
	connect(view_save_Action, SIGNAL(triggered()), this, SLOT(view_save_Slot()));
	connect(view_show_new_Action, SIGNAL(triggered()), this, SLOT(view_show_new_Slot()));
	connect(view_new_scatter_Action, SIGNAL(triggered()), this, SLOT(view_new_scatter_Slot()));

	connect(selection_all_Action, SIGNAL(triggered()), this, SLOT(selection_all_Slot()));
	connect(selection_none_Action, SIGNAL(triggered()), this, SLOT(selection_none_Slot()));
	connect(selection_inverse_Action, SIGNAL(triggered()), this, SLOT(selection_inverse_Slot()));

	connect(set_color_Action, SIGNAL(triggered()), this, SLOT(set_color_Slot()));

	//connect(commit_selection_in_current_layer_Action, SIGNAL(triggered()), this, SLOT(commit_selection_in_current_layer_Slot()));
	connect(commit_selection_to_new_layer_Action, SIGNAL(triggered()), this, SLOT(commit_selection_to_new_layer_Slot()));

	connect(axes_editor_Action, SIGNAL(triggered()), this, SLOT(axes_editor_Slot()));//
	connect(axes_mode_Action, SIGNAL(triggered()), this, SLOT(axes_mode_Slot()));
	connect(axes_display_edges_Action, SIGNAL(triggered()), this, SLOT(axes_display_edges_Slot()));

	connect(lines_display_unselected_Action, SIGNAL(triggered()), this, SLOT(lines_display_unselected_Slot()));
	connect(lines_display_unselected_listing_Action, SIGNAL(triggered()), this, SLOT(lines_display_unselected_listing_Slot()));
	connect(lines_display_unselected_GLview_Action, SIGNAL(triggered()), this, SLOT(lines_display_unselected_GLview_Slot()));
	connect(lines_display_zombies_Action, SIGNAL(triggered()), this, SLOT(lines_display_zombies_Slot()));
	connect(lines_display_zombies_listing_Action, SIGNAL(triggered()), this, SLOT(lines_display_zombies_listing_Slot()));
	connect(lines_display_zombies_GLview_Action, SIGNAL(triggered()), this, SLOT(lines_display_zombies_GLview_Slot()));
        
        connect(file_format_builder_Action, SIGNAL(triggered()), this, SLOT(file_format_builder_Slot()));

	//connect(whats_this_Action, SIGNAL(triggered()), this, SLOT(whats_this_Slot()));
	connect(about_Action, SIGNAL(triggered()), this, SLOT(about_Slot()));
	
	
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::connect_widgets()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::connect_widgets()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);
	connect(pv_ListingsTabWidget, SIGNAL(currentChanged(int)), this, SLOT(change_of_current_view_Slot()));

	/* for the this::color_changed_Signal() */
	connect(this, SIGNAL(color_changed_Signal()), this, SLOT(refresh_current_view_Slot()));
// FIXME: connect this elsewhere please! connect(this, SIGNAL(color_changed_Signal()),  pv_ListingWindow, SLOT(refresh_listing_Slot()));
	
	/* for this::selection_changed_Signal() */
	connect(this, SIGNAL(selection_changed_Signal()), this, SLOT(refresh_current_view_Slot()));
// FIXME really, there's should be a better place to connect this signal to. connect(this, SIGNAL(selection_changed_Signal()), pv_ListingWindow, SLOT(refresh_listing_Slot()));

	
}

/******************************************************************************
 *
 * Callback: filtering_function_foreach; Create one menu entry in filter per plugin
 *
 *****************************************************************************/
#if 0 // FIXME
void filtering_function_foreach(char *name, picviz_filter_t * /*filter*/, void *userdata)
{
	QAction *action;
	PVMainWindow *mw = reinterpret_cast<PVMainWindow *>(userdata);
	QMenu *menu = mw->filter_Menu;

	QString filter_name = QString(name);
	QString action_name = QString(name);
	action_name.replace('_', ' ');
	action_name[0] = action_name[0].toUpper();

	action = new QAction(action_name, menu);
	action->setObjectName(filter_name);
	mw->connect(action, SIGNAL(triggered()), mw, SLOT(filter_Slot()));

	menu->addAction(action);
}
#endif

// Check if we have already a menu with this name at this level
static QMenu *create_filters_menu_exists(QHash<QMenu *, int> actions_list, QString name, int level)
{
	QHashIterator<QMenu *, int> iter(actions_list);
	while (iter.hasNext()) {
		iter.next();
		QString menu_title = iter.key()->title();
		int menu_level = iter.value();

		if ((!menu_title.compare(name)) && (menu_level == level)) {
			return iter.key();
		}
	}

	return NULL;
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::create_filters_menu_and_actions()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::create_filters_menu_and_actions()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);

	QMenu *menu = filter_Menu;
	QHash<QMenu *, int> actions_list; // key = action name; value = menu level; Foo/Bar/Camp makes Foo at level 0, Bar at level 1, etc.

	LIB_FILTER(Picviz::PVLayerFilter) &filters_layer = 	LIB_FILTER(Picviz::PVLayerFilter)::get();
	LIB_FILTER(Picviz::PVLayerFilter)::list_filters const& lf = filters_layer.get_list();
	
	LIB_FILTER(Picviz::PVLayerFilter)::list_filters::const_iterator it;

	for (it = lf.begin(); it != lf.end(); it++) {
		//(*it).get_args()["Menu_name"]
		QString filter_name = QString(it.key());
		QString action_name = QString(it.key());
		QString status_tip = it.value()->status_bar_description();

		QStringList actions_name = action_name.split(QString("/"));
		if (actions_name.count() > 1) {
			// // qDebug("actions_name[0]=%s\n", qPrintable(actions_name[0]));
			// // We add the various submenus
			for (int i = 0; i < actions_name.count(); i++) {
				bool is_last = i == actions_name.count() - 1 ? 1 : 0;

				// Step 1: we add the different menus into the hash
				QMenu *menu_exists = create_filters_menu_exists(actions_list, actions_name[i], i);
				if (!menu_exists) {
					QMenu *filter_element_menu = new QMenu(actions_name[i]);
					actions_list[filter_element_menu] = i;
				}

				// Step 2: we connect the menus with each other and connect the actions
				QMenu *menu_to_add = create_filters_menu_exists(actions_list, actions_name[i], i);
				if (!menu_to_add) {
					PVLOG_ERROR("The menu named '%s' at position level %d cannot be added since it was not append previously!\n", qPrintable(actions_name[i]), i);
				}
				if (i == 0) { // We are at root level
					menu->addMenu(menu_to_add);
				} else {
					if (is_last) {
						QMenu *previous_menu = create_filters_menu_exists(actions_list, actions_name[i - 1], i - 1);
						
						QAction* action = new QAction(actions_name[i], previous_menu);
						action->setObjectName(filter_name);
						action->setStatusTip(status_tip);
						connect(action, SIGNAL(triggered()), this, SLOT(filter_Slot()));
						previous_menu->addAction(action);
					} else {
						// we add a menu to the previous menu
						QMenu *previous_menu = create_filters_menu_exists(actions_list, actions_name[i - 1], i - 1);
						QMenu *current_menu = create_filters_menu_exists(actions_list, actions_name[i], i);
						previous_menu->addMenu(current_menu);
					}
				}
			}
		} else {	// Nothing to split, so there is only a direct action
			QAction* action = new QAction(action_name, menu);
			action->setObjectName(filter_name);
			action->setStatusTip(status_tip);
			connect(action, SIGNAL(triggered()), this, SLOT(filter_Slot()));

			menu->addAction(action);
		}
	}
}

void PVInspector::PVMainWindow::import_type(PVRush::PVInputType_p in_t)
{
	// PVRush::PVInputType_p in_t = PVInputTypeMenuEntries::input_type_from_action((QAction*) sender());
	// PVRush::PVInputType_p in_t = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("file");
	PVRush::list_creators lcr = PVRush::PVSourceCreatorFactory::get_by_input_type(in_t);
	PVRush::hash_format_creator format_creator = PVRush::PVSourceCreatorFactory::get_supported_formats(lcr);

	PVRush::hash_formats formats;

	PVRush::hash_format_creator::const_iterator itfc;
	for (itfc = format_creator.begin(); itfc != format_creator.end(); itfc++) {
		formats[itfc.key()] = itfc.value().first;
	}

	// Create the input widget
	QString choosenFormat;
	// PVInputType::list_inputs is a QList<QVariant>
	PVRush::PVInputType::list_inputs inputs;
	map_files_types files_multi_formats;
	QHash<QString,PVRush::PVInputType::input_type> hash_input_name;

	if (!in_t->createWidget(formats, inputs, choosenFormat, this))
		return; // This means that the user pressed the "cancel" button
	
	QHash< QString,PVRush::PVInputType::list_inputs > discovered;
	QHash<QString,PVCore::PVMeanValue<float> > discovered_types; // format->mean_success_rate

	bool file_type_found = false;

	if (choosenFormat.compare(PICVIZ_AUTOMATIC_FORMAT_STR) == 0) {
		PVRush::PVInputType::list_inputs::const_iterator itin;

		// Go through the inputs
		for (itin = inputs.begin(); itin != inputs.end(); itin++) {
			QString in_str = (*itin).toString();
			hash_input_name[in_str] = *itin;

			// Pre-discovery to have some sources already eliminated and
			// save the custom formats of the remaining sources
			PVRush::list_creators::const_iterator itcr;
			PVRush::list_creators pre_discovered_creators;
			PVRush::hash_formats custom_formats;
			for (itcr = lcr.begin(); itcr != lcr.end(); itcr++) {
				PVRush::PVSourceCreator_p sc = *itcr;
				if (sc->pre_discovery(*itin)) {
					pre_discovered_creators.push_back(sc);
					in_t->get_custom_formats(*itin, custom_formats);
				}
			}

			// Load possible formats of the remaining sources
			PVRush::hash_format_creator dis_format_creator = PVRush::PVSourceCreatorFactory::get_supported_formats(pre_discovered_creators);

			// Add the custom formats
			PVRush::hash_formats::const_iterator it_cus_f;
			for (it_cus_f = custom_formats.begin(); it_cus_f != custom_formats.end(); it_cus_f++) {
				// Save this custom format to the global formats object
				formats.insert(it_cus_f.key(), it_cus_f.value());

				PVRush::list_creators::const_iterator it_lc;
				for (it_lc = lcr.begin(); it_lc != lcr.end(); it_lc++) {
					PVRush::hash_format_creator::mapped_type v(it_cus_f.value(), *it_lc);
					dis_format_creator[it_cus_f.key()] = v;

					// Save this format/creator pair to the "format_creator" object
					format_creator[it_cus_f.key()] = v;
				}
			}

			// Try every possible format
			QHash<QString,PVCore::PVMeanValue<float> > file_types;
			try {
				for (itfc = dis_format_creator.begin(); itfc != dis_format_creator.end(); itfc++) {
					try {
						float success_rate = PVRush::PVSourceCreatorFactory::discover_input(itfc.value(), *itin);
						PVLOG_INFO("For input %s with format %s, success rate is %0.4f\n", qPrintable(in_str), qPrintable(itfc.key()), success_rate);
						if (success_rate > 0) {
							QString const& str_format = itfc.key();
							file_types[str_format].push(success_rate);
							discovered_types[str_format].push(success_rate);
						}
					}
					catch (PVRush::PVXmlParamParserException &e) {
						PVLOG_ERROR("Format XML parser error: %s\n", qPrintable(e.what()));
						continue;
					}
				}
			}
			catch (PVRush::PVInputException &e) {
				PVLOG_ERROR("PVInput error: %s\n", e.what().c_str());
				continue;
			}

			if (file_types.count() == 1) {
				// We got the formats that match this input
				discovered[file_types.keys()[0]].push_back(*itin);
			}
			else
			if (file_types.count() > 1) {
				files_multi_formats[in_str] = file_types.keys();
			}
		}

		file_type_found = (discovered.size() > 0) | (files_multi_formats.size() > 0);
	}
	else
	{
		file_type_found = true;
		discovered[choosenFormat] = inputs;
	}

	if (!file_type_found) {
		QMessageBox msgBox;
		msgBox.critical(this, "Cannot import file", "The file cannot be opened: invalid file or type!\nReasons can be:\n  * PCAP with no IP packets\n  * PCAP with Netflow without SYN packets (uncheck default Netflow in options)\n  * Invalid parser providing no results\n");
		PVLOG_ERROR("Cannot import source!\n");
		return;
	}

	if (discovered_types.size() > 1) {
		QStringList dis_types = discovered_types.keys();
		QStringList dis_types_comment;
		QList<PVCore::PVMeanValue<float> > rates = discovered_types.values();
		QList<PVCore::PVMeanValue<float> >::const_iterator itf;
		for (itf = rates.begin(); itf != rates.end(); itf++) {
			dis_types_comment << QString("mean success rate = %1%").arg(itf->compute_mean()*100);
		}

		PVStringListChooserWidget *choosew = new PVStringListChooserWidget(this, "Multiple types have been detected.\nPlease choose the one(s) you need and press OK.", dis_types, dis_types_comment);
		if (!choosew->exec())
			return;
		QStringList sel_types = choosew->get_sel_list();

		QStringList to_remove;
		if (dis_types.size() != sel_types.size()) {
			// Remove types that are not in dis_types from 'discovered'
			for (int i = 0; i < dis_types.size(); i++) {
				QString const& t_ = dis_types[i];	
				if (sel_types.contains(t_))
					continue;
				discovered.remove(t_);
				to_remove << t_;
			}
		}

		// Remove the types to remove from files_multi_types
		map_files_types::iterator it = files_multi_formats.begin();
		while (it != files_multi_formats.end()) {
			QStringList& types_ = (*it).second;
			for (int i = 0; i < to_remove.size(); i++) {
				types_.removeOne(to_remove[i]);
			}
			if (types_.size() == 1) {
				discovered[types_[0]] << (*it).first;
				map_files_types::iterator it_rem = it;
				it++;
				files_multi_formats.erase(it_rem);
			}
			else {
				it++;
			}
		}
	}
	
	if (files_multi_formats.size() > 0) {
		PVFilesTypesSelWidget* files_types_sel = new PVFilesTypesSelWidget(this, files_multi_formats);
		if (!files_types_sel->exec())
			return;
		// Add everything to the discovered table
		map_files_types::const_iterator it;
		for (it = files_multi_formats.begin(); it != files_multi_formats.end(); it++) {
			QStringList const& types_l = (*it).second;
			QString const& input_name = (*it).first;
			for (int i = 0; i < types_l.size(); i++) {
				discovered[types_l[i]] << hash_input_name[input_name];
			}
		}
	}

	Picviz::PVScene_p import_scene;
	Picviz::PVMapping_p import_mapping;
	Picviz::PVMapped_p import_mapped;
	Picviz::PVPlotting_p import_plotting;
	Picviz::PVPlotted_p import_plotted;
	Picviz::PVView_p import_view;
	PVGL::PVMessage message;

	// Load a type of file per view
	QHash< QString, PVRush::PVInputType::list_inputs >::const_iterator it = discovered.constBegin();
	for (; it != discovered.constEnd(); it++) {
		// Create scene and source
		import_scene = Picviz::PVScene_p(new Picviz::PVScene(const_cast<char*>("default"), root)); // FIXME!
		Picviz::PVSource_p import_source = Picviz::PVSource_p(new Picviz::PVSource(import_scene));

		const PVRush::PVInputType::list_inputs& inputs = it.value();
		const QString& type = it.key();

		PVRush::pair_format_creator const& fc = format_creator[type];

		// AG: the tab index is a mix of the directory and type
		// If there is only one file, its filename is used instead of the directory
		QString tab_name = in_t->tab_name_of_inputs(inputs);
		tab_name += QString(" / ")+type;

		PVRush::PVControllerJob_p job_import;
		try {
			job_import = import_source->files_append(fc.first, fc.second, inputs);
		}
		catch (PVRush::PVInputException &e) {
			PVLOG_ERROR("PVInput error: %s\n", e.what().c_str());
			continue;
		}

		if (!PVExtractorWidget::show_job_progress_bar(job_import, job_import->nb_elts_max(), this)) {
			job_import->cancel();
			message.function = PVGL_COM_FUNCTION_DESTROY_TRANSIENT;
			pvgl_com->post_message_to_gl(message);
			continue;
		}
		job_import->wait_end();
		PVLOG_INFO("The normalization job took %0.4f seconds.\n", job_import->duration().seconds());
		if (import_source->nraw->table.size() == 0) {
			PVLOG_ERROR("Cannot append source!\n");
			QMessageBox msgBox;
			msgBox.critical(this, "Cannot import file type", QString("The files %1 cannot be opened. It looks like the format is invalid (invalid regular expressions or filters).").arg(tab_name));
			message.function = PVGL_COM_FUNCTION_DESTROY_TRANSIENT;
			pvgl_com->post_message_to_gl(message);
			continue;
		}
		import_source->set_limits(pv_ImportFileDialog->from_line_edit->text().toUInt(), pv_ImportFileDialog->to_line_edit->text().toUInt());
		import_source->get_extractor().dump_nraw();

#ifndef CUDA
		// Transient view. This need to be created before posting the "PVGL_COM_FUNCTION_CREATE_VIEW" message,
		// because the actual GL view is created by this message. Cf. libpvgl/src/PVMain.cpp::timer_func
		// for more informations.
		message.function = PVGL_COM_FUNCTION_PLEASE_WAIT;
		message.pointer_1 = new QString(tab_name);
		pvgl_com->post_message_to_gl(message);
#endif

		import_mapping = Picviz::PVMapping_p(new Picviz::PVMapping(import_source));
		import_mapped = Picviz::PVMapped_p(new Picviz::PVMapped(import_mapping));
		import_plotting = Picviz::PVPlotting_p(new Picviz::PVPlotting(import_mapped));
		import_plotted = Picviz::PVPlotted_p(new Picviz::PVPlotted(import_plotting));
		import_view = Picviz::PVView_p(new Picviz::PVView(import_plotted));
		import_view->process_from_layer_stack();

		current_tab = new PVTabSplitter(this, import_view, tab_name, pv_ListingsTabWidget);
		if(current_tab!=0)
                    connect(current_tab,SIGNAL(selection_changed_signal(bool)),this,SLOT(enable_menu_filter_Slot(bool)));
#ifdef CUDA
		// Transient view. This need to be created before posting the "PVGL_COM_FUNCTION_CREATE_VIEW" message,
		// because the actual GL view is created by this message. Cf. libpvgl/src/PVMain.cpp::timer_func
		// for more informations.
		message.function = PVGL_COM_FUNCTION_PLEASE_WAIT;
		message.pointer_1 = new QString(tab_name);
		pvgl_com->post_message_to_gl(message);
#endif
		
		// Ask the PVGL to create a GL-View from the previous transient view
		message.function = PVGL_COM_FUNCTION_CREATE_VIEW;
		message.pv_view = import_view;
		pvgl_com->post_message_to_gl(message);
		int new_tab_index = pv_ListingsTabWidget->addTab(current_tab, tab_name);
		/* Set the new tab as the active tab */
		pv_ListingsTabWidget->setCurrentIndex(new_tab_index);
	}

	if (discovered.size() > 0) {
		menu_activate_is_file_opened(true);
	}

	
	pv_labelWelcomeIcon->hide();
	pv_ImportFileButton->hide();
	pv_ListingsTabWidget->setVisible(true);
}

void PVInspector::PVMainWindow::display_icon_Slot()
{
	pv_labelWelcomeIcon->setVisible(true);
	pv_ImportFileButton->setVisible(true);
}

void PVInspector::PVMainWindow::import_type_default_Slot()
{
	import_type(LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("file"));
}


void PVInspector::PVMainWindow::import_type_Slot()
{
	QAction* action_src = (QAction*) sender();
	QString const& itype = action_src->data().toString();
	PVRush::PVInputType_p in_t = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name(itype);
	import_type(in_t);	
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::create_pvgl_thread
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::create_pvgl_thread ()
{
	pvgl_thread = new PVGL::PVThread ();
	pvgl_com = pvgl_thread->get_com();
	pvgl_thread->start ();
	timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(check_messages()));
	timer->start(100);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::destroy_pvgl_views
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::destroy_pvgl_views(Picviz::PVView_p view)
{
	PVGL::PVMessage message;

	message.function = PVGL_COM_FUNCTION_DESTROY_VIEWS;
	message.pv_view = view;
	pvgl_com->post_message_to_gl(message);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::lines_display_unselected_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::lines_display_unselected_Slot()
{
	Picviz::PVView_p current_lib_view;
	Picviz::PVStateMachine *state_machine = NULL;

	if (!current_tab) {
		return;
	}
	current_lib_view = current_tab->get_lib_view();
	state_machine = current_lib_view->state_machine;

	if (pv_ListingsTabWidget->currentIndex() == -1) {
		return;
	}

	state_machine->toggle_gl_unselected_visibility();
	state_machine->toggle_listing_unselected_visibility();
	/* We set the listing to be the same */
	// state_machine->set_listing_unselected_visibility(state_machine->are_unselected_visible());//???
	/* We refresh the view */
	current_lib_view->process_visibility();
	update_pvglview(current_lib_view, PVGL_COM_REFRESH_SELECTION);
	/* We refresh the listing */
	current_tab->update_pv_listing_model_Slot();

	if (!lines_display_unselected_Action->text().compare(QString(tr("Hide unselected lines")))) {
		lines_display_unselected_Action->setText(QString(tr("Display unselected lines")));
	} else {
		lines_display_unselected_Action->setText(QString(tr("Hide unselected lines")));
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::eventFilter
 *
 *****************************************************************************/
bool PVInspector::PVMainWindow::eventFilter(QObject *watched_object, QEvent *event)
{
	//PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);

	if (watched_object == pv_ListingsTabWidget->get_tabBar()) {
		if (event->type() == QEvent::KeyPress) {
			QKeyEvent *temp_keyEvent = static_cast<QKeyEvent*>(event);
			if ((temp_keyEvent->key() == Qt::Key_Left) || (temp_keyEvent->key() == Qt::Key_Right)) {
				keyPressEvent(temp_keyEvent);
				return true;
			} else {
				return false;
			}
		} else {
			return false;
		}
	} else {
		// pass the event on to the parent class
		return QMainWindow::eventFilter(watched_object, event);
	}
}


/******************************************************************************
 *
 * PVInspector::PVMainWindow::keyPressEvent()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::keyPressEvent(QKeyEvent *event)
{
	/* VARIABLES */
	int column_index;
	int number_of_selected_lines;
	/* We prepare a direct access to the current lib_view */
	Picviz::PVView_p current_lib_view;
	/* ... and the current_selected_layer */
	Picviz::PVLayer *current_selected_layer = NULL;
	/* We also need an access to the state machine */
	Picviz::PVStateMachine *state_machine = NULL;
	/* things needed for the screenshot */
	QString initial_path;
	QImage screenshot_image;
	QPixmap screenshot_pixmap;
	QString screenshot_filename;

	if (current_tab) {
		current_lib_view = current_tab->get_lib_view();
		state_machine = current_lib_view->state_machine;
	}
	/* Now we switch according to the key pressed */
	switch (event->key()) {

		/* Select all */
		case Qt::Key_A:
			/* If there is no view at all, don't do anything */
			if (pv_ListingsTabWidget->currentIndex() == -1) {
				break;
			}
			switch (event->modifiers()) {
				case (Qt::ShiftModifier):
					current_lib_view->floating_selection.select_all();
					break;

				default:
					current_lib_view->volatile_selection = current_lib_view->layer_stack_output_layer.get_selection();
//						current_lib_view->layer_stack_output_layer.get_selection().A2B_copy(current_lib_view->volatile_selection);
					break;
			}

			/* We deactivate the square area */
			state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_OFF);
			/* We process the view from the selection */
			current_lib_view->process_from_selection();
			/* We refresh the view */
			update_pvglview(current_lib_view, PVGL_COM_REFRESH_SELECTION);
			/* We refresh the listing */
			current_tab->refresh_listing_with_horizontal_header_Slot();
			current_tab->update_pv_listing_model_Slot();
			current_tab->refresh_listing_Slot();
			break;

		case Qt::Key_C:
			/* If there is no view at all, don't do anything */
			if (pv_ListingsTabWidget->currentIndex() == -1) {
				break;
			}
			set_color_Slot();
			break;

		case Qt::Key_D:
			// current_lib_view->layer_stack.write_file("out.data");
			break;
		case Qt::Key_E:
			// current_lib_view->layer_stack.read_file("out.data");
			break;

		/* Delete active axis */
		case Qt::Key_Delete:
			/* If there is no view at all, don't do anything */
			if (pv_ListingsTabWidget->currentIndex() == -1) {
				break;
			}
			/* If we are not in AXIS_MODE, don't do anything */
			if (!state_machine->is_axes_mode()) {
				break;
			}

			/* We decide to leave at least two axes... */
			if (current_lib_view->get_axes_count() <= 2) {
				break;
			}

			switch (event->modifiers()) {
				case (Qt::MetaModifier):

						break;

				case (Qt::AltModifier):

						break;

				case (Qt::ShiftModifier):

						break;

				default:
					current_lib_view->axes_combination.remove_axis(current_lib_view->active_axis);
					/* We check if we have just removed the rightmost axis */
					if ( current_lib_view->axes_combination.get_axes_count() == current_lib_view->active_axis ) {
						current_lib_view->active_axis -= 1;
					}
					break;
			}

			update_pvglview(current_lib_view, PVGL_COM_REFRESH_POSITIONS|PVGL_COM_REFRESH_AXES);
			current_tab->refresh_listing_with_horizontal_header_Slot();
			current_tab->update_pv_listing_model_Slot();
			current_tab->refresh_listing_Slot();
			break;


		/* How much ? : Gives the number of selected lines */
		case Qt::Key_Dollar:
			if (pv_ListingsTabWidget->currentIndex() == -1) {
				break;
			}
			number_of_selected_lines = current_lib_view->get_number_of_selected_lines();
			PVLOG_ERROR(" There is now %d selected lines \n", number_of_selected_lines);
			break;


		/* Decrease active axis column index */
		case Qt::Key_Down:
			/* If there is no view at all, don't do anything */
			if (pv_ListingsTabWidget->currentIndex() == -1) {
				break;
			}
			/* If we are not in AXIS_MODE, don't do anything */
			if (!state_machine->is_axes_mode()) {
				break;
			}

			/* We test if we have reached the lowest column_index value */
			column_index = current_lib_view->axes_combination.get_axis_column_index(current_lib_view->active_axis);
			if ( column_index <= 0 ) {
				break;
			}

			switch (event->modifiers()) {
				case (Qt::MetaModifier):
				case (Qt::AltModifier):
				case (Qt::ShiftModifier):
						break;

				default:
						current_lib_view->axes_combination.decrease_axis_column_index(current_lib_view->active_axis);
						break;
			}

			update_pvglview(current_lib_view, PVGL_COM_REFRESH_POSITIONS);
			current_tab->refresh_listing_with_horizontal_header_Slot();
			break;

		/* Forget about the current selection */
		case Qt::Key_Escape:
			/* If there is no view at all, don't do anything */
			if (pv_ListingsTabWidget->currentIndex() == -1) {
				break;
			}
			/* We turn SQUARE AREA mode OFF */
			state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_OFF);
			/* We need to process the view from the selection */
			current_lib_view->process_from_selection();
			/* THEN we can refresh the view */
			update_pvglview(current_tab->get_lib_view(), PVGL_COM_REFRESH_SELECTION);
			//refresh_current_view_Slot();
			current_tab->refresh_listing_Slot();
			break;

		/* Kommit the current selection to an old/new layer */
		case Qt::Key_K:
			/* If there is no view at all, don't do anything */
			if (pv_ListingsTabWidget->currentIndex() == -1) {
				break;
			}

			switch (event->modifiers()) {
				case (Qt::MetaModifier): // The "Windows key!"
						/* We Kommit and restet to active layer the actuel output lines properties and selection */
						/* We get the current selected layer */
						current_selected_layer = &(current_lib_view->layer_stack.get_selected_layer());
						/* We fill it's lines_properties */
						current_lib_view->output_layer.get_lines_properties().A2B_copy_restricted_by_selection_and_nelts(current_selected_layer->get_lines_properties(), current_lib_view->real_output_selection, current_lib_view->row_count);
						// picviz_lines_properties_A2B_copy_restricted_by_selection_and_nelts(current_lib_view->output_layer.get_lines_properties(), current_selected_layer->lines_properties, current_lib_view->real_output_selection, current_lib_view->row_count);
						/* We fill it's selection */
						current_selected_layer->get_selection() = current_lib_view->real_output_selection;
						//current_lib_view->real_output_selection.A2B_copy(current_selected_layer.get_selection());
						/* We need to process the view from the layer_stack */
						current_lib_view->process_from_layer_stack();
						/* THEN we can refresh the view */
						update_pvglview(current_tab->get_lib_view(), PVGL_COM_REFRESH_SELECTION);
						current_tab->refresh_listing_Slot();
						PVLOG_INFO("%s: MetaModifier!\n", __FUNCTION__);
						break;

						/* We Kommit to a new layer */
				case (Qt::AltModifier):
						commit_selection_to_new_layer_Slot();
						//PVLOG_INFO("%s: AltModifier!\n", __FUNCTION__);
						break;

				case (Qt::ShiftModifier):
						/* We Kommit to active layer and add lines if not yet present */

						break;

						/* We Kommit to active layer (only the lines properties)*/
				default:
						commit_selection_in_current_layer_Slot();
						break;
			}

			break;

		/* Move active axis to the left */
		case Qt::Key_Left: // FIXME: should we keep this in the Qt view.
			/* If there is no view at all, don't do anything */
			if (pv_ListingsTabWidget->currentIndex() == -1) {
				break;
			}
			/* We test if we are at the leftmost axis */
			if (current_lib_view->active_axis == 0 ) {
				break;
			}

			switch (event->modifiers()) {
				case (Qt::MetaModifier):

						break;

				case (Qt::AltModifier):

						break;

				case (Qt::ShiftModifier):
						current_lib_view->axes_combination.move_axis_left_one_position(current_lib_view->active_axis);
						current_lib_view->active_axis -= 1;
						break;

				default:
						current_lib_view->active_axis -= 1;
						break;
			}

			update_pvglview(current_lib_view, PVGL_COM_REFRESH_POSITIONS);
			current_tab->refresh_listing_with_horizontal_header_Slot();
			break;


		/* Suppress the selected lines ... */
		case Qt::Key_Minus:
			/* If there is no view at all, don't do anything */
			if (pv_ListingsTabWidget->currentIndex() == -1) {
				break;
			}
			switch (event->modifiers()) {
				case (Qt::MetaModifier):

					break;

				case (Qt::AltModifier):


					break;

				case (Qt::ShiftModifier):
					/*  */

					break;

				/* We supress the actuel selected lines in the selected layer */
				default:
					/* We get the current selected layer */
					current_selected_layer = &(current_lib_view->layer_stack.get_selected_layer());
					/* We suppress the real_output_selection from it */
					current_selected_layer->get_selection() -= current_lib_view->real_output_selection;
					//current_selected_layer->get_selection().AB2A_substraction(current_lib_view->real_output_selection);
					/* We need to process the view from the layer_stack */
					current_lib_view->process_from_layer_stack();
					/* THEN we can emit the signal */
					//refresh_current_view_Slot();
					update_pvglview(current_tab->get_lib_view(), PVGL_COM_REFRESH_COLOR|PVGL_COM_REFRESH_ZOMBIES|PVGL_COM_REFRESH_SELECTION);
					current_tab->refresh_listing_Slot();
					break;
			}

			break;

				/* Toggle antialiasing */
		case Qt::Key_NumberSign:
				/* If there is no view at all, don't do anything */
				if (pv_ListingsTabWidget->currentIndex() == -1) {
					break;
				}
				/* We toggle the ANTIALIASING mode */
				state_machine->toggle_antialiased();
				/* We refresh the view */
				update_pvglview(current_lib_view, PVGL_COM_REFRESH_SELECTION|PVGL_COM_REFRESH_ZOMBIES);
				break;

		// This is only for testing purposes !
		case Qt::Key_Percent:
				// if (pv_ListingsTabWidget->currentIndex() == -1) {
				// 	break;
				// }

				// switch (event->modifiers()) {
				// 	case (Qt::ShiftModifier):
				// 		pv_AxisProperties->create();
				// 		pv_AxisProperties->show();
				// 			break;
				// }
				break;


				/* Add the selected lines ... */
		case Qt::Key_Plus:
				/* If there is no view at all, don't do anything */
				if (pv_ListingsTabWidget->currentIndex() == -1) {
					break;
				}
				switch (event->modifiers()) {
					case (Qt::MetaModifier):

							break;

					case (Qt::AltModifier):

							break;

					case (Qt::ShiftModifier):

							/* We add the actuel selected lines in the selected layer */
					case (Qt::NoModifier):
							Picviz::PVSelection temp_selection;
							Picviz::PVColor line_properties;
							// line_properties = picviz_line_properties_new();
							/* We get the current selected layer */
							current_selected_layer = &(current_lib_view->layer_stack.get_selected_layer());
							/* We compute the selection of lines really new to that layer */
							temp_selection = current_lib_view->real_output_selection - current_selected_layer->get_selection();
//							current_lib_view->real_output_selection.AB2C_substraction(current_selected_layer.get_selection(), temp_selection);
							/* We add the real_output_selection to the current selected layer */
							current_selected_layer->get_selection() -= current_lib_view->real_output_selection;
							//current_selected_layer.get_selection().AB2A_or(current_lib_view->real_output_selection);
							/* We set the line_properties of the newly added lines to default */
							current_selected_layer->get_lines_properties().A2A_set_to_line_properties_restricted_by_selection_and_nelts(line_properties, temp_selection, current_lib_view->row_count);
							// picviz_lines_properties_A2A_set_to_line_properties_restricted_by_selection_and_nelts(current_selected_layer.get_lines_properties(), line_properties, temp_selection, current_lib_view->row_count);
							/* We need to process the view from the layer_stack */
							current_lib_view->process_from_layer_stack();
							/* THEN we can emit the signal */

							update_pvglview(current_tab->get_lib_view(), PVGL_COM_REFRESH_SELECTION);
							current_tab->refresh_listing_Slot();

							break;
				}

				break;


				/* Move active axis to the right */
		case Qt::Key_Right:
				/* If there is no view at all, don't do anything */
				if (pv_ListingsTabWidget->currentIndex() == -1) {
					break;
				}
				/* We test if we are at the rightmost axis */
				if (current_lib_view->active_axis == current_lib_view->get_axes_count() - 1) {
					break;
				}

				switch (event->modifiers()) {
					case (Qt::MetaModifier):

							break;

					case (Qt::AltModifier):

							break;

					case (Qt::ShiftModifier):
							current_lib_view->axes_combination.move_axis_right_one_position(current_lib_view->active_axis);
							current_lib_view->active_axis += 1;
							break;

					default:
							current_lib_view->active_axis += 1;
							break;
				}

				update_pvglview(current_lib_view, PVGL_COM_REFRESH_SELECTION);
				current_tab->refresh_listing_with_horizontal_header_Slot();
				break;

				/* Make a screenshot and save it to a file */
		case Qt::Key_S:
				/* If there is no view at all, don't do anything */
				if (pv_ListingsTabWidget->currentIndex() == -1) {
					break;
				}
				initial_path = QDir::currentPath();
				switch (event->modifiers()) {
					/* We make a screenshot of the full desktop */
					case (Qt::AltModifier):
							screenshot_filename = pv_ListingsTabWidget->tabText(pv_ListingsTabWidget->currentIndex());
							screenshot_filename.append("_DESKTOP_%1.png");
							screenshot_filename = screenshot_filename.arg(current_tab->get_screenshot_index(), 3, 10, QString("0")[0]);
							current_tab->increment_screenshot_index();
							initial_path += "/" + screenshot_filename;

							screenshot_pixmap = QPixmap::grabWindow(QApplication::desktop()->winId());

							screenshot_filename = QFileDialog::getSaveFileName(this, tr("Save Screenshot As"), initial_path, tr("PNG Files (*.png);;All Files (*)"));
							if (!screenshot_filename.isEmpty()) {
								screenshot_pixmap.save(screenshot_filename, "png", 100);
							}
							break;

							/* We make a screenshot of the PV_MainWindow */
					case (Qt::ShiftModifier):
							screenshot_filename = pv_ListingsTabWidget->tabText(pv_ListingsTabWidget->currentIndex());
							screenshot_filename.append("_APP_%1.png");
							screenshot_filename = screenshot_filename.arg(current_tab->get_screenshot_index(), 3, 10, QString("0")[0]);
							current_tab->increment_screenshot_index();
							initial_path += "/" + screenshot_filename;

							screenshot_pixmap = QPixmap::grabWindow(this->winId());

							screenshot_filename = QFileDialog::getSaveFileName(this, tr("Save Screenshot As"), initial_path, tr("PNG Files (*.png);;All Files (*)"));
							if (!screenshot_filename.isEmpty()) {
								screenshot_pixmap.save(screenshot_filename, "png", 100);
							}
							break;

							/* We only make a screenshot of the PVGL Views */
					case (Qt::NoModifier):
								{
									QString *filename;
									screenshot_filename = pv_ListingsTabWidget->tabText(pv_ListingsTabWidget->currentIndex());
									screenshot_filename.append("_%1.png");
									screenshot_filename = screenshot_filename.arg(current_tab->get_screenshot_index(), 3, 10, QString("0")[0]);
									current_tab->increment_screenshot_index();
									initial_path += "/" + screenshot_filename;

									filename = new QString(QFileDialog::getSaveFileName(this, tr("Save Screenshot As"), initial_path, tr("PNG Files (*.png);;All Files (*)")));
									if (!filename->isEmpty()) {
										PVGL::PVMessage message;
										message.function = PVGL_COM_FUNCTION_TAKE_SCREENSHOT;
										message.pv_view = current_lib_view;
										message.int_1 = -1;
										message.pointer_1 = filename;
										pvgl_com->post_message_to_gl(message);
									}
								}
							break;
				}
				break;


				/* Toggle the menuBar visibility */
		case Qt::Key_Space:
				menuBar()->setVisible(! menuBar()->isVisible());
				break;

				/* toggle the visibility of the UNSELECTED lines */
		case Qt::Key_U:
				/* If there is no view at all, don't do anything */
				if (pv_ListingsTabWidget->currentIndex() == -1) {
					break;
				}
				switch (event->modifiers()) {
					/* We only toggle the Listing */
					case (Qt::AltModifier):
							/* We toggle*/
							state_machine->toggle_listing_unselected_visibility();
							/* We refresh the listing */
							current_tab->update_pv_listing_model_Slot();
							break;

							/* We only toggle the View */
					case (Qt::ShiftModifier):
							/* We toggle*/
							state_machine->toggle_gl_unselected_visibility();
							/* We refresh the view */
							current_lib_view->process_visibility();
							update_pvglview(current_lib_view, PVGL_COM_REFRESH_SELECTION);
							break;

							/* We toggle both the Listing and the View */
					default:
							/* We toggle the view first */
							state_machine->toggle_gl_unselected_visibility();
							/* We set the listing to be the same */
							state_machine->set_listing_unselected_visible(state_machine->are_gl_unselected_visible());
							/* We refresh the view */
							current_lib_view->process_visibility();
							update_pvglview(current_lib_view, PVGL_COM_REFRESH_SELECTION);
							/* We refresh the listing */
							current_tab->update_pv_listing_model_Slot();
							break;
				}
				break;


				/* Increase active axis column index */
		case Qt::Key_Up:
				/* If there is no view at all, don't do anything */
				if (pv_ListingsTabWidget->currentIndex() == -1) {
					break;
				}
				/* If we are not in AXIS_MODE, don(t do anything */
				if (!state_machine->is_axes_mode()) {
					break;
				}

				/* We test if we have reached the highest column_index value */
				column_index = current_lib_view->axes_combination.get_axis_column_index(current_lib_view->active_axis);
				if ( column_index >= current_lib_view->get_original_axes_count() - 1) {
					break;
				}

				switch (event->modifiers()) {
					case (Qt::MetaModifier):
					case (Qt::AltModifier):
					case (Qt::ShiftModifier):
							break;

					default:
						current_lib_view->axes_combination.increase_axis_column_index(current_lib_view->active_axis);
						break;
				}

				update_pvglview(current_lib_view, PVGL_COM_REFRESH_POSITIONS);
				current_tab->refresh_listing_with_horizontal_header_Slot();
				break;


				/* Toggle the AXES_MODE */
		case Qt::Key_X:
				/* If there is no view at all, don't do anything */
				if (pv_ListingsTabWidget->currentIndex() == -1) {
					break;
				}
				state_machine->toggle_axes_mode();

				/* if we enter in AXES_MODE we must disable SQUARE_AREA_MODE */
				if (state_machine->is_axes_mode()) {
					/* We turn SQUARE AREA mode OFF */
					state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_OFF);
				}

				update_pvglview(current_lib_view, PVGL_COM_REFRESH_SELECTION);
				current_tab->refresh_listing_Slot();
				break;

				/* Toggle the visibility of the ZOMBIE lines */
		case Qt::Key_Z:
				/* If there is no view at all, don't do anything */
				if (pv_ListingsTabWidget->currentIndex() == -1) {
					break;
				}

				switch (event->modifiers()) {
					/* We only toggle the Listing */
					case (Qt::AltModifier):
							/* We toggle */
							state_machine->toggle_listing_zombie_visibility();
							/* We refresh the listing */
							current_tab->update_pv_listing_model_Slot();
							break;

							/* We only toggle the View */
					case (Qt::ShiftModifier):
							/* We toggle */
							state_machine->toggle_gl_zombie_visibility();
							/* We refresh the view */
							current_lib_view->process_visibility();
							update_pvglview(current_lib_view, PVGL_COM_REFRESH_SELECTION);
							break;

							/* We toggle both the Listing and the View */
					default:
							/* We toggle the view first */
							state_machine->toggle_gl_zombie_visibility();
							/* We set the listing to be the same */
							state_machine->set_listing_zombie_visible(state_machine->are_gl_zombie_visible());
							/* We refresh the view */
							current_lib_view->process_visibility();
							update_pvglview(current_lib_view, PVGL_COM_REFRESH_SELECTION);
							/* We refresh the listing */
							current_tab->update_pv_listing_model_Slot();
							break;
				}
				break;
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::refresh_view()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::refresh_view(Picviz::PVView_p picviz_view)
{
	for (int i = 0; i < pv_ListingsTabWidget->count();i++) {
		PVTabSplitter *tab = dynamic_cast<PVTabSplitter*>(pv_ListingsTabWidget->widget(i));
		if (!tab) {
			PVLOG_ERROR("PVInspector::PVMainWindow::%s: Tab isn't tab!!!\n", __FUNCTION__);
		} else {
			if (tab->get_lib_view() == picviz_view) {
				/* We refresh the layerstack */
				tab->refresh_layer_stack_view_Slot();
				/* We refresh the view */
				update_pvglview(tab->get_lib_view(), PVGL_COM_REFRESH_SELECTION);
				/* We refresh the listing */
				tab->refresh_listing_Slot();
			}
		}
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::set_color()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::set_color(Picviz::PVView_p picviz_view)
{
	/* VARIABLES */
	unsigned char r;
	unsigned char g;
	unsigned char b;
	unsigned char a;
	QColor color;

	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);

	/* We let the user select a color */
	color = pv_ColorDialog->getColor(Qt::white, NULL, "test", QColorDialog::ShowAlphaChannel);
	/* We test if the user canceled the dialog */
	if ( ! color.isValid() ) {
		return;
	}

	/* The user DID select a color... */
	/* We get the color value */
	r = (unsigned char)color.red();
	g = (unsigned char)color.green();
	b = (unsigned char)color.blue();
	a = (unsigned char)color.alpha();

	// We don't allow a completly black color (it is reserved for zombie)
	if (r == 0 && b == 0 && g == 0) {
		r = 2;
	}
	/* We paint the lines in the post_filter_layer */
	picviz_view->set_color_on_post_filter_layer(r, g, b, a);
	//picviz_view->set_color_on_active_layer(r, g, b, a);
	/* We process the view from the EventLine */
	picviz_view->process_from_eventline();

	/* We refresh the view */
	update_pvglview(picviz_view, PVGL_COM_REFRESH_COLOR);
	//FIXME! This should be done in a function
	for (int i = 0; i < pv_ListingsTabWidget->count();i++) {
		PVTabSplitter *tab = dynamic_cast<PVTabSplitter*>(pv_ListingsTabWidget->widget(i));
		if (!tab) {
			PVLOG_ERROR("PVInspector::PVMainWindow::%s: Tab isn't tab!!!\n", __FUNCTION__);
		} else {
			if (tab->get_lib_view() == picviz_view) {
				/* We refresh the listing */
				tab->refresh_listing_Slot();
			}
		}
	}

	// And we commit to the current layer (cf. ticket #38)
	commit_selection_in_current_layer(current_tab->get_lib_view());
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::update_check
 *
 *****************************************************************************/
int PVInspector::PVMainWindow::update_check()
{
#ifdef CUSTOMER_RELEASE
	QNetworkAccessManager *manager = new QNetworkAccessManager(this);
	QNetworkRequest request;

	connect(manager, SIGNAL(finished(QNetworkReply*)),
		this, SLOT(update_reply_finished_Slot(QNetworkReply*)));

	request.setUrl(QUrl("http://www.picviz.com/update.html"));
	request.setRawHeader("User-Agent", "Mozilla/5.0.1.15");

	manager->get(request);

#endif

	return 0;
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::update_pvglview
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::update_pvglview(Picviz::PVView_p view, int refresh_states)
{
	PVGL::PVMessage message;

	message.function = PVGL_COM_FUNCTION_REFRESH_VIEW;
	message.pv_view = view;
	message.int_1 = refresh_states;
	pvgl_com->post_message_to_gl(message);
}
