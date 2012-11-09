/**
 * \file PVMainWindow.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <QApplication>
#include <QDesktopWidget>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFile>
#include <QFrame>
#include <QFuture>
#include <QFutureWatcher>
#include <QLine>
#include <QMenuBar>
#include <QMessageBox>
#include <QSplashScreen>
#include <QStatusBar>
#include <QVBoxLayout>

#include <PVMainWindow.h>
#include <PVExtractorWidget.h>
#include <PVStringListChooserWidget.h>

#include <pvguiqt/PVWorkspace.h>
#include <pvguiqt/PVListingView.h>

#include <pvkernel/core/PVRecentItemsManager.h>

#ifdef CUSTOMER_RELEASE
  #ifdef WIN32
    #include <winlicensesdk.h>
  #endif
#endif	// CUSTOMER_RELEASE

#include <pvkernel/core/general.h>
#include <pvkernel/core/debug.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVMeanValue.h>
#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/core/PVVersion.h>

#include <pvkernel/rush/PVFileDescription.h>

#include <pvkernel/widgets/PVColorDialog.h>

#include <picviz/general.h>
#include <picviz/arguments.h>
#include <picviz/PVSelection.h>
#include <picviz/PVMapping.h>
#include <picviz/PVPlotting.h>
#include <picviz/PVStateMachine.h>
#include <picviz/PVSource.h>

#include <pvhive/PVActor.h>
#include <pvhive/PVCallHelper.h>

#include <pvguiqt/PVListDisplayDlg.h>

#include <PVFormatBuilderWidget.h>

QFile *report_file;

/******************************************************************************
 *
 * PVInspector::PVMainWindow::PVMainWindow
 *
 *****************************************************************************/
Q_DECLARE_METATYPE(Picviz::PVSource*);

PVInspector::PVMainWindow::PVMainWindow(QWidget *parent):
	QMainWindow(parent),
	_load_project_dlg(this, tr("Load a project..."), QString(), PICVIZ_SCENE_ARCHIVE_FILTER ";;" ALL_FILES_FILTER),
	_root(new Picviz::PVRoot())
{
	setAttribute(Qt::WA_DeleteOnClose);
	setAcceptDrops(true);

	//_ad2g_mw = NULL;

	// SIZE STUFF
	// WARNING: nothing should be set here.
	
	// OBJECTNAME STUFF
	setObjectName("PVMainWindow");
	
	// SPLASH SCREEN : we create the Splash screen
	QSplashScreen splash(QPixmap(":/splash-screen"));

	// License validity test : it's a simple "time" check
	if (time(NULL) >= CUSTOMER_RELEASE_EXPIRATION_DATE) {
		exit(0);
	}
	
	//We can show the Splash Screen
	splash.show();

	//setWindowFlags(Qt::FramelessWindowHint);

	// FIXME
	//_start_screen_widget = new PVStartScreenWidget(this);
	
	// FONT stuff
	QFontDatabase pv_font_database;
	pv_font_database.addApplicationFont(QString(":/Jura-DemiBold.ttf"));
	pv_font_database.addApplicationFont(QString(":/OSP-DIN.ttf"));

	setGeometry(20,10, 1024, 900);
//	datatree = picviz_datatreerootitem_new();

	//import_source = NULL;
	report_started = false;
	report_image_index = 0;
	report_filename = NULL;

	//We activate all available Windows
	pv_ExportSelectionDialog = new PVExportSelectionDialog(this);
	pv_ExportSelectionDialog->hide();

	_projects_tab_widget = new PVGuiQt::PVProjectsTabWidget(get_root());
	_projects_tab_widget->show();
	connect(_projects_tab_widget, SIGNAL(workspace_dragged_outside(QWidget*)), this, SLOT(create_new_window_for_workspace(QWidget*)));
	connect(_projects_tab_widget, SIGNAL(new_project()), this, SLOT(project_new_Slot()));
	connect(_projects_tab_widget, SIGNAL(load_project()), this, SLOT(project_load_Slot()));
	connect(_projects_tab_widget, SIGNAL(load_project_from_path(const QString &)), this, SLOT(load_project(const QString &)));
	connect(_projects_tab_widget, SIGNAL(save_project()), this, SLOT(project_save_Slot()));
	connect(_projects_tab_widget, SIGNAL(load_source_from_description(PVRush::PVSourceDescription)), this, SLOT(load_source_from_description_Slot(PVRush::PVSourceDescription)));
	connect(_projects_tab_widget, SIGNAL(import_type(const QString &)), this, SLOT(import_type_Slot(const QString &)) );
	connect(_projects_tab_widget, SIGNAL(new_format()), this, SLOT(new_format_Slot()));
	connect(_projects_tab_widget, SIGNAL(load_format()), this, SLOT(open_format_Slot()));
	connect(_projects_tab_widget, SIGNAL(edit_format(const QString &)), this, SLOT(edit_format_Slot(const QString &)));
	connect(_projects_tab_widget, SIGNAL(is_empty()), this, SLOT(display_icon_Slot()) );

	// We display the PV Icon together with a button to import files
	pv_centralMainWidget = new QWidget();
	pv_centralMainWidget->setObjectName("pv_centralMainWidget_of_PVMainWindow");

	pv_mainLayout = new QVBoxLayout();
	pv_mainLayout->setSpacing(40);
	pv_mainLayout->setContentsMargins(0,0,0,0);

	pv_mainLayout->addWidget(_projects_tab_widget);

	pv_centralMainWidget->setLayout(pv_mainLayout);

	pv_centralWidget = new QStackedWidget();
	pv_centralWidget->addWidget(pv_centralMainWidget);
	pv_centralWidget->setCurrentWidget(pv_centralMainWidget);

	setCentralWidget(pv_centralWidget);

	_projects_tab_widget->setFocus(Qt::OtherFocusReason);


	// RemoteLogDialog = new QMainWindow(this, Qt::Dialog);
	// QObject::connect(RemoteLogDialog, SIGNAL(destroyed()), this, SLOT(hide()));

	//We populate all actions, menus and connect them
	create_actions();
	create_menus();
	connect_actions();
	connect_widgets();
	menu_activate_is_file_opened(false);
	
	statusBar();
	statemachine_label = new QLabel("");
	statusBar()->insertPermanentWidget(0, statemachine_label);

	//splash.finish(_start_screen_widget);

	// Center the main window
	QRect r = geometry();
	r.moveCenter(QApplication::desktop()->screenGeometry(this).center());
	setGeometry(r);

	// Load version informations
	_last_known_cur_release = pvconfig.value(PVCONFIG_LAST_KNOWN_CUR_RELEASE, PICVIZ_VERSION_INVALID).toUInt();
	_last_known_maj_release = pvconfig.value(PVCONFIG_LAST_KNOWN_MAJ_RELEASE, PICVIZ_VERSION_INVALID).toUInt();

	update_check();

	// The default title isn't set, so do this by hand...
	setWindowTitle(QString("Picviz Inspector " PICVIZ_CURRENT_VERSION_STR));


	//Set stylesheet
	QFile css_file(":/gui.css");
	css_file.open(QFile::ReadOnly);
	QTextStream css_stream(&css_file);
	QString css_string(css_stream.readAll());
	css_file.close();
	setStyleSheet(css_string);

	show();
}

bool PVInspector::PVMainWindow::event(QEvent* event)
{
	QString mime_type = "application/x-picviz_workspace";

	if(event->type() == QEvent::DragEnter) {
		QDragEnterEvent* dragEnterEvent = static_cast<QDragEnterEvent*>(event);
		if (dragEnterEvent->mimeData()->hasFormat(mime_type)) {
			dragEnterEvent->accept(); // dragEnterEvent->acceptProposedAction();
			return true;
		}
	}
	else if (event->type() == QEvent::Drop) {
		QDropEvent* dropEvent = static_cast<QDropEvent*>(event);
		if (dropEvent->mimeData()->hasFormat(mime_type)) {
			dropEvent->acceptProposedAction();
			const QMimeData* mimeData = dropEvent->mimeData();
			QByteArray byte_array = mimeData->data(mime_type);
			if (byte_array.size() < (int)sizeof(PVGuiQt::PVWorkspace*)) {
				return false;
			}
			PVGuiQt::PVWorkspace* workspace = *(reinterpret_cast<PVGuiQt::PVWorkspace* const*>(byte_array.constData()));

			_projects_tab_widget->add_workspace(workspace);

			return true;
		}
	}

	return QMainWindow::event(event);
}

// These two methods are intentionally put in PVMainWindow's implementation
// as this might change in the near future and save lots of compilation time.
Picviz::PVRoot& PVInspector::PVMainWindow::get_root()
{
	return *_root;
}

Picviz::PVRoot_sp PVInspector::PVMainWindow::get_root_sp()
{
	return _root;
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::auto_detect_formats
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::auto_detect_formats(PVFormatDetectCtxt ctxt)
{
	PVRush::PVInputType::list_inputs::const_iterator itin;

	// Go through the inputs
	for (itin = ctxt.inputs.begin(); itin != ctxt.inputs.end(); itin++) {
		QString in_str = (*itin)->human_name();
		ctxt.hash_input_name[in_str] = *itin;

		// Pre-discovery to have some sources already eliminated and
		// save the custom formats of the remaining sources
		PVRush::list_creators::const_iterator itcr;
		PVRush::list_creators pre_discovered_creators;
		PVRush::hash_formats custom_formats;
		for (itcr = ctxt.lcr.begin(); itcr != ctxt.lcr.end(); itcr++) {
			PVRush::PVSourceCreator_p sc = *itcr;
			if (sc->pre_discovery(*itin)) {
				pre_discovered_creators.push_back(sc);
				ctxt.in_t->get_custom_formats(*itin, custom_formats);
			}
		}

		// Load possible formats of the remaining sources
		PVRush::hash_format_creator dis_format_creator = PVRush::PVSourceCreatorFactory::get_supported_formats(pre_discovered_creators);

		// Add the custom formats
		PVRush::hash_formats::const_iterator it_cus_f;
		for (it_cus_f = custom_formats.begin(); it_cus_f != custom_formats.end(); it_cus_f++) {
			// Save this custom format to the global formats object
			ctxt.formats.insert(it_cus_f.key(), it_cus_f.value());

			PVRush::list_creators::const_iterator it_lc;
			for (it_lc = ctxt.lcr.begin(); it_lc != ctxt.lcr.end(); it_lc++) {
				PVRush::hash_format_creator::mapped_type v(it_cus_f.value(), *it_lc);
				dis_format_creator[it_cus_f.key()] = v;

				// Save this format/creator pair to the "format_creator" object
				ctxt.format_creator[it_cus_f.key()] = v;
			}
		}

		// Try every possible format
		QHash<QString,PVCore::PVMeanValue<float> > file_types;
		tbb::tick_count dis_start = tbb::tick_count::now();

		QList<PVRush::hash_format_creator::key_type> dis_formats = dis_format_creator.keys();
		QList<PVRush::hash_format_creator::mapped_type> dis_v = dis_format_creator.values();
		bool input_exception = false;
		std::string input_exception_str;
#pragma omp parallel for
		for (int i = 0; i < dis_format_creator.size(); i++) {
			//PVRush::pair_format_creator const& pfc = itfc.value();
			PVRush::pair_format_creator const& pfc = dis_v.at(i);
			//QString const& str_format = itfc.key();
			QString const& str_format = dis_formats.at(i);
			try {
				float success_rate = PVRush::PVSourceCreatorFactory::discover_input(pfc, *itin);

				if (success_rate > 0) {
#pragma omp critical
					{
						PVLOG_INFO("For input %s with format %s, success rate is %0.4f\n", qPrintable(in_str), qPrintable(str_format), success_rate);
						file_types[str_format].push(success_rate);
						ctxt.discovered_types[str_format].push(success_rate);
					}
				}
			}
			catch (PVRush::PVXmlParamParserException &e) {
#pragma omp critical
				{
					ctxt.formats_error[pfc.first.get_full_path()] = std::pair<QString,QString>(pfc.first.get_format_name(), tr("XML parser error: ") + e.what());
				}
				continue;
			}
			catch (PVRush::PVFormatInvalid &e) {
#pragma omp critical
				{
					ctxt.formats_error[pfc.first.get_full_path()] = std::pair<QString,QString>(pfc.first.get_format_name(), e.what());
				}
				continue;
			}
			catch (PVRush::PVInputException &e)
			{
#pragma omp critical
				{
					input_exception = true;
					input_exception_str = e.what().c_str();
				}
			}
		}
		tbb::tick_count dis_end = tbb::tick_count::now();
		PVLOG_INFO("Automatic format discovery took %0.4f seconds.\n", (dis_end-dis_start).seconds());
		if (input_exception) {
			PVLOG_ERROR("PVInput exception: %s\n", input_exception_str.c_str());
			continue;
		}

		if (file_types.count() == 1) {
			// We got the formats that matches this input
			ctxt.discovered[file_types.keys()[0]].push_back(*itin);
		}
		else {
			if (file_types.count() > 1) {
				ctxt.files_multi_formats[in_str] = file_types.keys();
			}
		}
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::closeEvent
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::closeEvent(QCloseEvent* event)
{
	if (_projects_tab_widget->save_modified_projects()) {
		PVCore::PVProgressBox* pbox = new PVCore::PVProgressBox(tr("Closing Picviz Inspector..."), (QWidget*) this);
		pbox->set_enable_cancel(false);
		PVCore::PVProgressBox::progress(boost::bind(&PVMainWindow::close_all_views, this), pbox);
		event->accept();
	}
	else {
		event->ignore();
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::close_all_views
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::close_all_views()
{
	close_scene();
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::close_scene
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::close_scene()
{
	// Close sources one by one
	/*int ntabs = _workspaces_tab_widget->count();
	for (int i = 0; i < ntabs; i++) {
		_workspaces_tab_widget->remove_workspace(0);
	}
	if (_ad2g_mw) {
		_ad2g_mw->deleteLater();
	}
	_scene = PVCore::PVDataTreeAutoShared<Picviz::PVScene>(get_root_sp(), "default");
	_workspaces_tab_widget->set_scene(_scene.get());
	_ad2g_mw = NULL;
	set_project_modified(false);*/
}


/******************************************************************************
 *
 * PVInspector::PVMainWindow::commit_selection_in_current_layer
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::commit_selection_in_current_layer(Picviz::PVView* picviz_view)
{
	//Picviz::StateMachine *state_machine = NULL;

	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);

	//state_machine = picviz_view->state_machine;

	/* We get the current selected layer */
	Picviz::PVLayer &current_selected_layer = picviz_view->layer_stack.get_selected_layer();
	/* We fill it's lines_properties */
	picviz_view->output_layer.get_lines_properties().A2B_copy_restricted_by_selection_and_nelts(current_selected_layer.get_lines_properties(), picviz_view->real_output_selection, picviz_view->row_count);

	/* We need to process the view from the layer_stack */
	Picviz::PVView_sp view_sp = picviz_view->shared_from_this();
	PVHive::PVCallHelper::call<FUNC(Picviz::PVView::process_from_layer_stack)>(view_sp);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::commit_selection_to_new_layer
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::commit_selection_to_new_layer(Picviz::PVView* picviz_view)
{
	// Register an actor to the hive
	Picviz::PVView_sp view_sp = picviz_view->shared_from_this();
	PVHive::PVActor<Picviz::PVView> actor;
	PVHive::get().register_actor(view_sp, actor);

	actor.call<FUNC(Picviz::PVView::add_new_layer)>();
	Picviz::PVLayer &new_layer = picviz_view->layer_stack.get_selected_layer();
	/* We set it's selection to the final selection */
	picviz_view->set_selection_with_final_selection(new_layer.get_selection());
	picviz_view->output_layer.get_lines_properties().A2B_copy_restricted_by_selection_and_nelts(new_layer.get_lines_properties(), new_layer.get_selection(), picviz_view->row_count);
	
	/* We need to reprocess the layer stack */
	new_layer.compute_min_max(*picviz_view->get_parent<Picviz::PVPlotted>());

	actor.call<FUNC(Picviz::PVView::process_from_layer_stack)>();
}
/******************************************************************************
 *
 * PVInspector::PVMainWindow::move_selection_to_new_layer
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::move_selection_to_new_layer(Picviz::PVView* picviz_view)
{
	// Register an actor to the hive
	Picviz::PVView_sp view_sp = picviz_view->shared_from_this();
	PVHive::PVActor<Picviz::PVView> actor;
	PVHive::get().register_actor(view_sp, actor);

	Picviz::PVLayer& current_layer = picviz_view->layer_stack.get_selected_layer();
	actor.call<FUNC(Picviz::PVView::add_new_layer)>();
	Picviz::PVLayer &new_layer = picviz_view->layer_stack.get_selected_layer();
	/* We set it's selection to the final selection */
	picviz_view->set_selection_with_final_selection(new_layer.get_selection());
	picviz_view->output_layer.get_lines_properties().A2B_copy_restricted_by_selection_and_nelts(new_layer.get_lines_properties(), new_layer.get_selection(), picviz_view->row_count);
	// We remove that selection from the current layer
	current_layer.get_selection().and_not(new_layer.get_selection());
	
	/* We need to reprocess the layer stack */
	new_layer.compute_min_max(*picviz_view->get_parent<Picviz::PVPlotted>());

	actor.call<FUNC(Picviz::PVView::process_from_layer_stack)>();
}


/******************************************************************************
 *
 * PVInspector::PVMainWindow::connect_widgets()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::connect_widgets()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);
	//connect(_projects_tab_widget, SIGNAL(currentChanged(int)), this, SLOT(change_of_current_view_Slot()));

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
 * PVInspector::PVMainWindow::create_filters_menu_and_actions
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::create_filters_menu_and_actions()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);

	QMenu *menu = filter_Menu;
	QHash<QMenu *, int> actions_list; // key = action name; value = menu level; Foo/Bar/Camp makes Foo at level 0, Bar at level 1, etc.

	LIB_CLASS(Picviz::PVLayerFilter) &filters_layer = 	LIB_CLASS(Picviz::PVLayerFilter)::get();
	LIB_CLASS(Picviz::PVLayerFilter)::list_classes const& lf = filters_layer.get_list();
	LIB_CLASS(Picviz::PVLayerFilter)::list_classes::const_iterator it;

	for (it = lf.begin(); it != lf.end(); it++) {
		//(*it).get_args()["Menu_name"]
		QString filter_name = it.key();
		QString action_name = it.value()->menu_name();
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

/******************************************************************************
 *
 * PVInspector::PVMainWindow::display_icon_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::display_icon_Slot()
{
	menu_activate_is_file_opened(false);
}


/******************************************************************************
 *
 * PVInspector::PVMainWindow::get_tab_from_view
 *
 *****************************************************************************/
PVGuiQt::PVWorkspace* PVInspector::PVMainWindow::get_tab_from_view(Picviz::PVView* picviz_view)
{
	return get_tab_from_view(*picviz_view);
}

PVGuiQt::PVWorkspace* PVInspector::PVMainWindow::get_tab_from_view(Picviz::PVView const& picviz_view)
{
	// This returns the tab associated to a picviz view
	const Picviz::PVScene* scene = picviz_view.get_parent<Picviz::PVScene>();
	PVGuiQt::PVSceneWorkspacesTabWidget* workspaces_tab_widget = _projects_tab_widget->get_workspace_tab_widget_from_scene(scene);
	for (int i = 0; workspaces_tab_widget && i < workspaces_tab_widget->count();i++) {
		PVGuiQt::PVWorkspace *tab = dynamic_cast<PVGuiQt::PVWorkspace*>(workspaces_tab_widget->widget(i));
		if (!tab) {
			PVLOG_ERROR("PVInspector::PVMainWindow::%s: Tab isn't tab!!!\n", __FUNCTION__);
		} else {
			if (tab->current_view() == &picviz_view) {
				return tab;
				/* We refresh the listing */
			}
		}
	}
	return nullptr;
}



/******************************************************************************
 *
 * PVInspector::PVMainWindow::import_type
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::import_type(PVRush::PVInputType_p in_t)
{
	PVRush::list_creators lcr = PVRush::PVSourceCreatorFactory::get_by_input_type(in_t);
	PVRush::hash_format_creator format_creator = PVRush::PVSourceCreatorFactory::get_supported_formats(lcr);

	PVRush::hash_formats formats, new_formats;

	PVRush::hash_format_creator::const_iterator itfc;
	for (itfc = format_creator.begin(); itfc != format_creator.end(); itfc++) {
		formats[itfc.key()] = itfc.value().first;
	}

	// Create the input widget
	QString choosenFormat;
	// PVInputType::list_inputs is a QList<PVInputDescription_p>
	PVRush::PVInputType::list_inputs inputs;

	PVCore::PVArgumentList args_extract = PVRush::PVExtractor::default_args_extractor();

	if (!in_t->createWidget(formats, new_formats, inputs, choosenFormat, args_extract, this))
		return; // This means that the user pressed the "cancel" button

	// Add the new formats to the formats
	{
		PVRush::hash_formats::iterator it;
		for (it = new_formats.begin(); it != new_formats.end(); it++) {
			formats[it.key()] = it.value();
			PVRush::list_creators::const_iterator it_lc;
			for (it_lc = lcr.begin(); it_lc != lcr.end(); it_lc++) {
				PVRush::hash_format_creator::mapped_type v(it.value(), *it_lc);
				// Save this format/creator pair to the "format_creator" object
				format_creator[it.key()] = v;
			}
		}
	}

	import_type(in_t, inputs, formats, format_creator, choosenFormat, args_extract);
}


/******************************************************************************
 *
 * PVInspector::PVMainWindow::import_type
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::import_type(PVRush::PVInputType_p in_t, PVRush::PVInputType::list_inputs const& inputs, PVRush::hash_formats& formats, PVRush::hash_format_creator& format_creator, QString const& choosenFormat, PVCore::PVArgumentList const& args_ext)
{
	PVRush::list_creators lcr = PVRush::PVSourceCreatorFactory::get_by_input_type(in_t);

	QHash< QString,PVRush::PVInputType::list_inputs > discovered;
	QHash<QString,PVCore::PVMeanValue<float> > discovered_types; // format->mean_success_rate

	QHash<QString, std::pair<QString,QString> > formats_error; // Errors w/ some formats

	map_files_types files_multi_formats;
	QHash<QString,PVRush::PVInputDescription_p> hash_input_name;

	bool file_type_found = false;

	if (choosenFormat.compare(PICVIZ_AUTOMATIC_FORMAT_STR) == 0) {
		PVCore::PVProgressBox* pbox = new PVCore::PVProgressBox(tr("Auto-detecting file format..."), (QWidget*) this);
		pbox->set_enable_cancel(false);
		if (!PVCore::PVProgressBox::progress(boost::bind(&PVMainWindow::auto_detect_formats, this, PVFormatDetectCtxt(inputs, hash_input_name, formats, format_creator, files_multi_formats, discovered, formats_error, lcr, in_t, discovered_types)), pbox)) {
			return;
		}
		file_type_found = (discovered.size() > 0) | (files_multi_formats.size() > 0);
	}
	else
	{
		file_type_found = true;
		discovered[choosenFormat] = inputs;
	}

	treat_invalid_formats(formats_error);
	
	if (!file_type_found) {
		QString msg = "<p>The sources cannot be opened: automatic format detection reported <strong>no valid format</strong>.</p>";
		msg += "<p>Please note that automatic format detection is only applied on a small subset of the provided sources.</p>";
		msg += "<p><strong>Trick:</strong> if you know the format of these sources, and if it contains one or more filters that invalidate a lot of elements, you should avoid automatic format detection and select this format by hand in the import sources dialog.</p>";
		QMessageBox::warning(this, "Cannot import sources", msg);
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
				discovered[types_[0]] << hash_input_name[(*it).first];
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

	bool one_extraction_successful = false;
	bool save_inv_elts = args_ext["inv_elts"].toBool();
	// Load a type of file per view
	QHash< QString, PVRush::PVInputType::list_inputs >::const_iterator it = discovered.constBegin();
	for (; it != discovered.constEnd(); it++) {
		// Create scene and source

		const PVRush::PVInputType::list_inputs& inputs = it.value();
		const QString& type = it.key();

		PVRush::pair_format_creator const& fc = format_creator[type];

		PVRush::PVControllerJob_p job_import;
		PVRush::PVFormat const& cur_format = fc.first;

		Picviz::PVSource_sp import_source;
		try {
			if (_projects_tab_widget->projects_count() == 0) {
				project_new_Slot();
			}
			PVRush::PVSourceDescription src_desc(inputs, fc.second, cur_format);
			Picviz::PVScene_p scene_p = current_scene()->shared_from_this();
			import_source = PVHive::call<FUNC(Picviz::PVScene::add_source_from_description)>(scene_p, src_desc);
			import_source->set_invalid_elts_mode(save_inv_elts);
		}
		catch (PVRush::PVFormatException const& e) {
			PVLOG_ERROR("Error with format: %s\n", qPrintable(e.what()));
			continue;
		}

		if (!load_source(import_source)) {
			continue;
		}
		//import_source->set_parent(_scene);
		//_scene->add_source(import_source);

		one_extraction_successful = true;
	}

	if (!one_extraction_successful) {
		return;
	}

	menu_activate_is_file_opened(true);
	show_start_page(false);
	_projects_tab_widget->setVisible(true);
}



/******************************************************************************
 *
 * PVInspector::PVMainWindow::import_type_default_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::import_type_default_Slot()
{
	import_type(LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("file"));
}



/******************************************************************************
 *
 * PVInspector::PVMainWindow::import_type_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::import_type_Slot()
{
	QAction* action_src = (QAction*) sender();
	QString const& itype = action_src->data().toString();
	import_type_Slot(itype);
}

void PVInspector::PVMainWindow::import_type_Slot(const QString & itype)
{
	PVRush::PVInputType_p in_t = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name(itype);
	import_type(in_t);
}



/******************************************************************************
 *
 * PVInspector::PVMainWindow::keyPressEvent()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::keyPressEvent(QKeyEvent *event)
{
	QMainWindow::keyPressEvent(event);
	PVLOG_INFO("keyPressEvent\n");
#ifdef PICVIZ_DEVELOPER_MODE
	switch (event->key()) {

		case Qt::Key_Dollar:
		{
			/*if (pv_WorkspacesTabWidget->currentIndex() == -1) {
				break;
			}*/
			PVLOG_INFO("Reloading CSS\n");

			QFile css_file(PICVIZ_SOURCE_DIRECTORY "/gui-qt/src/resources/gui.css");
			if (css_file.open(QFile::ReadOnly)) {
				QTextStream css_stream(&css_file);
				QString css_string(css_stream.readAll());
				css_file.close();

				setStyleSheet(css_string);
				setStyle(QApplication::style());
			}
			break;
		}
	}
#endif

#if 0
	/* VARIABLES */
	int column_index;
	/* We prepare a direct access to the current lib_view */
	Picviz::PVView* current_lib_view;
	/* ... and the current_selected_layer */
	Picviz::PVLayer *current_selected_layer = NULL;
	/* We also need an access to the state machine */
	Picviz::PVStateMachine *state_machine = NULL;
	/* things needed for the screenshot */
	QString initial_path;
	QImage screenshot_image;
	QPixmap screenshot_pixmap;
	QString screenshot_filename;
	
	// FIXME!  This is so UGLY !!!
	QFile css_file("/donnees/GIT/OLD/picviz-inspector/gui-qt/src/resources/gui.css");
	css_file.open(QFile::ReadOnly);
	QTextStream css_stream(&css_file);
	QString css_string(css_stream.readAll());
	css_file.close();
	


	if (current_tab) {
		current_lib_view = current_tab->current_view();
		state_machine = current_lib_view->state_machine;
	}
	/* Now we switch according to the key pressed */
	switch (event->key()) {

		/* Select all */
		case Qt::Key_A:
			/* If there is no view at all, don't do anything */
			if (_workspaces_tab_widget->currentIndex() == -1) {
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
			//update_pvglview(current_lib_view, PVSDK_MESSENGER_REFRESH_SELECTION);
			/* We refresh the listing */
			current_tab->refresh_listing_with_horizontal_header_Slot();
			current_tab->update_pv_listing_model_Slot();
			current_tab->refresh_listing_Slot();
			break;

		case Qt::Key_C:
			/* If there is no view at all, don't do anything */
			if (_workspaces_tab_widget->currentIndex() == -1) {
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
			if (_workspaces_tab_widget->currentIndex() == -1) {
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

			//update_pvglview(current_lib_view, PVSDK_MESSENGER_REFRESH_POSITIONS|PVSDK_MESSENGER_REFRESH_AXES);
			current_tab->refresh_listing_with_horizontal_header_Slot();
			current_tab->update_pv_listing_model_Slot();
			current_tab->refresh_listing_Slot();
			break;


#ifndef NDEBUG
		case Qt::Key_Dollar:
		{
			if (_workspaces_tab_widget->currentIndex() == -1) {
				break;
			}

			// PhS
			setStyleSheet(css_string);
			setStyle(QApplication::style());

			QFile css_file("/donnees/GIT/OLD/picviz-inspector/gui-qt/src/resources/gui.css");
			css_file.open(QFile::ReadOnly);
			QTextStream css_stream(&css_file);
			QString css_string(css_stream.readAll());
			css_file.close();

			// PhS
			setStyleSheet(css_string);
			setStyle(QApplication::style());
			break;
		}
#endif

		/* Decrease active axis column index */
		case Qt::Key_Down:
			/* If there is no view at all, don't do anything */
			if (_workspaces_tab_widget->currentIndex() == -1) {
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

			//update_pvglview(current_lib_view, PVSDK_MESSENGER_REFRESH_POSITIONS);
			current_tab->refresh_listing_with_horizontal_header_Slot();
			break;

		/* Forget about the current selection */
		case Qt::Key_Escape:
			/* If there is no view at all, don't do anything */
			if (_workspaces_tab_widget->currentIndex() == -1) {
				break;
			}
			/* We turn SQUARE AREA mode OFF */
			state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_OFF);
			/* We need to process the view from the selection */
			current_lib_view->process_from_selection();
			/* THEN we can refresh the view */
			//update_pvglview(current_tab->get_lib_view(), PVSDK_MESSENGER_REFRESH_SELECTION);
			//refresh_current_view_Slot();
			current_tab->refresh_listing_Slot();
			break;

		/* Kommit the current selection to an old/new layer */
		case Qt::Key_K:
			/* If there is no view at all, don't do anything */
			if (_workspaces_tab_widget->currentIndex() == -1) {
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
						//update_pvglview(current_tab->get_lib_view(), PVSDK_MESSENGER_REFRESH_SELECTION);
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
			if (_workspaces_tab_widget->currentIndex() == -1) {
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

			//update_pvglview(current_lib_view, PVSDK_MESSENGER_REFRESH_POSITIONS);
			current_tab->refresh_listing_with_horizontal_header_Slot();
			break;


		/* Suppress the selected lines ... */
		case Qt::Key_Minus:
			/* If there is no view at all, don't do anything */
			if (_workspaces_tab_widget->currentIndex() == -1) {
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
					//update_pvglview(current_tab->get_lib_view(), PVSDK_MESSENGER_REFRESH_COLOR|PVSDK_MESSENGER_REFRESH_ZOMBIES|PVSDK_MESSENGER_REFRESH_SELECTION);
					current_tab->refresh_listing_Slot();
					break;
			}

			break;
 
		case Qt::Key_Enter:
		case Qt::Key_Return: {
			if (current_tab) {
				current_tab->pv_listing_view->keyEnterPressed();
			}
			break;
		}

				/* Toggle antialiasing */
		case Qt::Key_NumberSign:
				/* If there is no view at all, don't do anything */
				if (_workspaces_tab_widget->currentIndex() == -1) {
					break;
				}
				/* We toggle the ANTIALIASING mode */
				state_machine->toggle_antialiased();
				/* We refresh the view */
				//update_pvglview(current_lib_view, PVSDK_MESSENGER_REFRESH_SELECTION|PVSDK_MESSENGER_REFRESH_ZOMBIES);
				break;

		// This is only for testing purposes !
		case Qt::Key_Percent:
				// if (pv_WorkspacesTabWidget->currentIndex() == -1) {
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
				if (_workspaces_tab_widget->currentIndex() == -1) {
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
							PVCore::PVColor line_properties;
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

							//update_pvglview(current_tab->get_lib_view(), PVSDK_MESSENGER_REFRESH_SELECTION);
							current_tab->refresh_listing_Slot();

							break;
				}

				break;


				/* Move active axis to the right */
		case Qt::Key_Right:
				/* If there is no view at all, don't do anything */
				if (_workspaces_tab_widget->currentIndex() == -1) {
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

				//update_pvglview(current_lib_view, PVSDK_MESSENGER_REFRESH_SELECTION);
				current_tab->refresh_listing_with_horizontal_header_Slot();
				break;

				/* Make a screenshot and save it to a file */
		case Qt::Key_S:
				/* If there is no view at all, don't do anything */
				if (_workspaces_tab_widget->currentIndex() == -1) {
					break;
				}
				initial_path = QDir::currentPath();
				switch (event->modifiers()) {
					/* We make a screenshot of the full desktop */
					case (Qt::AltModifier):
							screenshot_filename = _workspaces_tab_widget->tabText(_workspaces_tab_widget->currentIndex());
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
							screenshot_filename = _workspaces_tab_widget->tabText(_workspaces_tab_widget->currentIndex());
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
									screenshot_filename = _workspaces_tab_widget->tabText(_workspaces_tab_widget->currentIndex());
									screenshot_filename.append("_%1.png");
									screenshot_filename = screenshot_filename.arg(current_tab->get_screenshot_index(), 3, 10, QString("0")[0]);
									current_tab->increment_screenshot_index();
									initial_path += "/" + screenshot_filename;

									filename = new QString(QFileDialog::getSaveFileName(this, tr("Save Screenshot As"), initial_path, tr("PNG Files (*.png);;All Files (*)")));
									if (!filename->isEmpty()) {
										PVSDK::PVMessage message;
										message.function = PVSDK_MESSENGER_FUNCTION_TAKE_SCREENSHOT;
										message.pv_view = current_lib_view->shared_from_this();
										message.int_1 = -1;
										message.pointer_1 = filename;
										pvsdk_messenger->post_message_to_gl(message);
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
		case Qt::Key_U:	// FIXME: U is useless and it taken by the menu
				// /* If there is no view at all, don't do anything */
				// if (pv_WorkspacesTabWidget->currentIndex() == -1) {
				// 	break;
				// }
				// switch (event->modifiers()) {
				// 	/* We only toggle the Listing */
				// 	case (Qt::AltModifier):
				// 			/* We toggle*/
				// 			state_machine->toggle_listing_unselected_visibility();
				// 			/* We refresh the listing */
				// 			current_tab->update_pv_listing_model_Slot();
				// 			break;

				// 			/* We only toggle the View */
				// 	case (Qt::ShiftModifier):
				// 			/* We toggle*/
				// 			state_machine->toggle_gl_unselected_visibility();
				// 			/* We refresh the view */
				// 			current_lib_view->process_visibility();
				// 			//update_pvglview(current_lib_view, PVSDK_MESSENGER_REFRESH_SELECTION);
				// 			break;

				// 			/* We toggle both the Listing and the View */
				// 	default:
				// 			/* We toggle the view first */
				// 			state_machine->toggle_gl_unselected_visibility();
				// 			/* We set the listing to be the same */
				// 			state_machine->set_listing_unselected_visible(state_machine->are_gl_unselected_visible());
				// 			/* We refresh the view */
				// 			current_lib_view->process_visibility();
				// 			//update_pvglview(current_lib_view, PVSDK_MESSENGER_REFRESH_SELECTION);
				// 			/* We refresh the listing */
				// 			current_tab->update_pv_listing_model_Slot();
				// 			break;
				// }
				break;


				/* Increase active axis column index */
		case Qt::Key_Up:
				/* If there is no view at all, don't do anything */
				if (_workspaces_tab_widget->currentIndex() == -1) {
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

				//update_pvglviecurrent_tabw(current_lib_view, PVSDK_MESSENGER_REFRESH_POSITIONS);
				current_tab->refresh_listing_with_horizontal_header_Slot();
				break;


				/* Toggle the AXES_MODE */
		case Qt::Key_X:
				/* If there is no view at all, don't do anything */
				if (_workspaces_tab_widget->currentIndex() == -1) {
					break;
				}
				state_machine->toggle_axes_mode();

				/* if we enter in AXES_MODE we must disable SQUARE_AREA_MODE */
				if (state_machine->is_axes_mode()) {
					/* We turn SQUARE AREA mode OFF */
					state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_OFF);
				}

				//update_pvglview(current_lib_view, PVSDK_MESSENGER_REFRESH_SELECTION);
				current_tab->refresh_listing_Slot();
				break;
	}
#endif
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::lines_display_unselected_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::lines_display_unselected_Slot()
{

	Picviz::PVStateMachine *state_machine = NULL;

	if (!current_view()) {
		return;
	}
	state_machine = current_view()->state_machine;

	state_machine->toggle_gl_unselected_visibility();
	state_machine->toggle_listing_unselected_visibility();
}


/******************************************************************************
 *
 * PVInspector::PVMainWindow::load_files
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::load_files(std::vector<QString> const& files, QString format)
{
	if (files.size() == 0) {
		return;
	}

	PVRush::PVInputType_p in_file = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("file");
	PVRush::list_creators lcr = PVRush::PVSourceCreatorFactory::get_by_input_type(in_file);
	PVRush::hash_format_creator format_creator = PVRush::PVSourceCreatorFactory::get_supported_formats(lcr);

	PVRush::hash_formats formats;
	{
		PVRush::hash_format_creator::const_iterator itfc;
		for (itfc = format_creator.begin(); itfc != format_creator.end(); itfc++) {
			formats[itfc.key()] = itfc.value().first;
		}
	}

	// Create PVFileDescription objects
	//
	
	PVRush::PVInputType::list_inputs files_in;
	{
		std::vector<QString>::const_iterator it;
		for (it = files.begin(); it != files.end(); it++) {
			files_in.push_back(PVRush::PVInputDescription_p(new PVRush::PVFileDescription(*it)));
		}
	}
	
	if (!format.isEmpty()) {
		PVRush::PVFormat new_format("custom:arg", format);
		formats["custom:arg"] = new_format;

		PVRush::list_creators::const_iterator it_lc;
		for (it_lc = lcr.begin(); it_lc != lcr.end(); it_lc++) {
			PVRush::hash_format_creator::mapped_type v(new_format, *it_lc);
			// Save this format/creator pair to the "format_creator" object
			format_creator["custom:arg"] = v;
		}
		format = "custom:arg";
	}
	else {
		format = PICVIZ_AUTOMATIC_FORMAT_STR;
	}

	import_type(in_file, files_in, formats, format_creator, format, PVRush::PVExtractor::default_args_extractor());
}



/******************************************************************************
 *
 * PVInspector::PVMainWindow::load_scene
 *
 *****************************************************************************/
bool PVInspector::PVMainWindow::load_scene(Picviz::PVScene* scene)
{
	// Here, load the whole scene.
	for (auto source_p : scene->get_children<Picviz::PVSource>()) {
		if (!load_source(source_p)) {
			return false;
		}
	}

	return true;
}

void PVInspector::PVMainWindow::display_inv_elts()
{
	if (current_view()) {
		if (current_view()->get_parent<Picviz::PVSource>()->get_invalid_elts().size() > 0) {
			//tab_src->get_source_invalid_elts_dlg()->show();
		}
		else {
			QMessageBox::information(this, tr("Invalid elements"), tr("No invalid element have been saved or created during the extraction of this source."));
		}
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::load_source
 *
 *****************************************************************************/
bool PVInspector::PVMainWindow::load_source(Picviz::PVSource_sp src)
{
	PVLOG_INFO("load_source: %p\n", src.get());
	// Load a created source

	if (src->get_children<Picviz::PVMapped>().size() == 0) {
		Picviz::PVMapped_p default_mapped(src);
	}

	// Extract the source
	PVRush::PVControllerJob_p job_import;
	try {
		job_import = src->extract();
	}
	catch (PVRush::PVInputException &e) {
		PVLOG_ERROR("PVInput error: %s\n", e.what().c_str());
		return false;
	}

	if (!PVExtractorWidget::show_job_progress_bar(job_import, src->get_format_name(), job_import->nb_elts_max(), this)) {
		//message.function = PVSDK_MESSENGER_FUNCTION_DESTROY_TRANSIENT;
		//pvsdk_messenger->post_message_to_gl(message);
		return false;
	}
	src->wait_extract_end(job_import);
	PVLOG_INFO("The normalization job took %0.4f seconds.\n", job_import->duration().seconds());
	if (src->get_rushnraw().get_number_rows() == 0) {
		QString msg = QString("<p>The files <strong>%1</strong> using format <strong>%2</strong> cannot be opened. ").arg(src->get_name()).arg(src->get_format_name());
		PVRow nelts = job_import->rejected_elements();
		if (nelts > 0) {
			msg += QString("Indeed, <strong>%1 elements</strong> have been extracted but were <strong>all invalid</strong>.</p>").arg(nelts);
			msg += QString("<p>This is because one or more splitters and/or filters defined in format <strong>%1</strong> reported invalid elements during the extraction.<br />").arg(src->get_format_name());
			msg += QString("You may have invalid regular expressions set in this format, or simply all the lines have been invalidated by one or more filters thus no lines matches your criterias.</p>");
			msg += QString("<p>You might try to <strong>fix your format</strong> or try to load <strong>another set of data</strong>.</p>");
		}
		else {
			msg += QString("Indeed, the sources <strong>were empty</strong> (empty files, bad database query, etc...) because no elements have been extracted.</p><p>You should try to load another set of data.</p>");
		}
		//message.function = PVSDK_MESSENGER_FUNCTION_DESTROY_TRANSIENT;
		//pvsdk_messenger->post_message_to_gl(message);
		QMessageBox::warning(this, "Cannot load sources", msg);
		return false;
	}
	src->get_extractor().dump_nraw();

	// If no view is present, create a default one. Otherwise, process them by
	// keeping the existing layers !
	bool success = true;
	if (src->get_children<Picviz::PVView>().size() == 0) {
		if (!PVCore::PVProgressBox::progress(boost::bind<void>(&Picviz::PVSource::create_default_view, src.get()), tr("Processing..."), (QWidget*) this)) {
			success = false;
		}
	}
	else {
		if (!PVCore::PVProgressBox::progress(boost::bind(&Picviz::PVSource::process_from_source, src.get()), tr("Processing..."), (QWidget*) this)) {
			success = false;
		}
	}

	if (!success) {
		return false;
	}

	_projects_tab_widget->add_source(src.get());

	if (src->get_children<Picviz::PVView>().size() > 0) {
		Picviz::PVView_sp first_view_p = src->get_children<Picviz::PVView>().at(0);
		first_view_p->get_parent<Picviz::PVRoot>()->select_view(*first_view_p);
	}

	//connect(current_tab,SIGNAL(selection_changed_signal(bool)),this,SLOT(enable_menu_filter_Slot(bool)));

	//_projects_tab_widget->setCurrentIndex(new_tab_index);

	if (src->get_invalid_elts().size() > 0) {
		display_inv_elts();
	}

	PVHive::call<FUNC(PVCore::PVRecentItemsManager::add)>(PVCore::PVRecentItemsManager::get(), src->get_format().get_full_path(), PVCore::PVRecentItemsManager::Category::USED_FORMATS);

	PVHive::call<FUNC(PVCore::PVRecentItemsManager::add_source)>(PVCore::PVRecentItemsManager::get(), src->get_source_creator(), src->get_inputs(), src->get_format());

	menu_activate_is_file_opened(true);
	return true;
}


/******************************************************************************
 *
 * PVInspector::PVMainWindow::set_color()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::set_color(Picviz::PVView* picviz_view)
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);

	/* We let the user select a color */
	PVWidgets::PVColorDialog* pv_ColorDialog = new PVWidgets::PVColorDialog(this);
	if (pv_ColorDialog->exec() != QDialog::Accepted) {
		return;
	}
	PVCore::PVHSVColor color = pv_ColorDialog->color();

	PVHive::PVActor<Picviz::PVView> actor;
	Picviz::PVView_sp view_sp = picviz_view->shared_from_this();
	PVHive::get().register_actor(view_sp, actor);

	//actor.call<FUNC(Picviz::PVView::set_color_on_post_filter_layer)>(color);
	actor.call<FUNC(Picviz::PVView::set_color_on_active_layer)>(color);
	actor.call<FUNC(Picviz::PVView::process_from_layer_stack)>();
	//commit_selection_in_current_layer(picviz_view);
}



/******************************************************************************
 *
 * PVInspector::PVMainWindow::set_color_selected
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::set_color_selected(QColor const& color)
{
#if 0
	Picviz::PVView& picviz_view = dlg->current_view();

	// Get the tab associated w/ this view
	PVTabSplitter* tab = get_tab_from_view(picviz_view);
	if (!tab) {
		return;
	}


	unsigned char r;
	unsigned char g;
	unsigned char b;
	unsigned char a;

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
	//FIXME: HSV window is needed !
	//picviz_view.set_color_on_post_filter_layer(r, g, b, a);
	//picviz_view->set_color_on_active_layer(r, g, b, a);
	/* We process the view from the EventLine */
	picviz_view.process_from_eventline();

	// And we commit to the current layer (cf. ticket #38)
	commit_selection_in_current_layer(current_tab->current_view());

	// And tell that the project has been modified
#endif
}



/******************************************************************************
 *
 * PVInspector::PVMainWindow::set_selection_from_layer
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::set_selection_from_layer(Picviz::PVView_sp view, Picviz::PVLayer const& layer)
{
	view->set_selection_from_layer(layer);
	//update_pvglview(view, PVSDK_MESSENGER_REFRESH_SELECTION);
}


/******************************************************************************
 *
 * PVInspector::PVMainWindow::set_version_informations
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::set_version_informations()
{
	if (_last_known_cur_release != PICVIZ_VERSION_INVALID) {
		pv_lastCurVersion->setText(PVCore::PVVersion::to_str(_last_known_cur_release));
	}
	if (_last_known_maj_release != PICVIZ_VERSION_INVALID) {
		pv_lastMajVersion->setText(PVCore::PVVersion::to_str(_last_known_maj_release));
	}
}



/******************************************************************************
 *
 * PVInspector::PVMainWindow::show_start_page
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::show_start_page(bool visible)
{
	/*if (visible) {
		pv_centralWidget->setCurrentWidget(pv_centralStartWidget);
	}
	else {
		pv_centralWidget->setCurrentWidget(pv_centralMainWidget);
	}*/
}



/******************************************************************************
 *
 * PVInspector::PVMainWindow::treat_invalid_formats
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::treat_invalid_formats(QHash<QString, std::pair<QString,QString> > const& errors)
{
	if (errors.size() == 0) {
		return;
	}

	if (!pvconfig.value(PVCONFIG_FORMATS_SHOW_INVALID, PVCONFIG_FORMATS_SHOW_INVALID_DEFAULT).toBool()) {
	   return;
	}

	// Get the current ignore list
	QStringList formats_ignored = pvconfig.value(PVCONFIG_FORMATS_INVALID_IGNORED, QStringList()).toStringList();

	// And remove them from the error list
	QHash<QString, std::pair<QString, QString> > errors_ = errors;
	for (int i = 0; i < formats_ignored.size(); i++) {
		errors_.remove(formats_ignored[i]);
	}

	if (errors_.size() == 0) {
		return;
	}

	QMessageBox msg(QMessageBox::Warning, tr("Invalid formats"), tr("Some formats were invalid."));
   	msg.setInformativeText(tr("You can simply ignore this message, choose not to display it again (for every format), or remove this warning only for these formats."));
	QPushButton* ignore = msg.addButton(QMessageBox::Ignore);
	msg.setDefaultButton(ignore);
	QPushButton* always_ignore = msg.addButton(tr("Always ignore these formats"), QMessageBox::AcceptRole);
	QPushButton* never_again = msg.addButton(tr("Never display this message again"), QMessageBox::RejectRole);

	QString detailed_txt;
	QHash<QString, std::pair<QString,QString> >::const_iterator it;
	for (it = errors_.begin(); it != errors_.end(); it++) {
		detailed_txt += it.value().first + QString(" (") + it.key() + QString("): ") + it.value().second + QString("\n");
	}
	msg.setDetailedText(detailed_txt);

	msg.exec();

	QPushButton* clicked_btn = (QPushButton*) msg.clickedButton();

	if (clicked_btn == ignore) {
		return;
	}

	if (clicked_btn == never_again) {
		pvconfig.setValue(PVCONFIG_FORMATS_SHOW_INVALID, QVariant(false));
		return;
	}

	if (clicked_btn == always_ignore) {
		// Append these formats to the ignore list
		formats_ignored.append(errors_.keys());
		pvconfig.setValue(PVCONFIG_FORMATS_INVALID_IGNORED, formats_ignored);
	}
}



/******************************************************************************
 *
 * PVInspector::PVMainWindow::update_check
 *
 *****************************************************************************/
int PVInspector::PVMainWindow::update_check()
{
#ifdef CUSTOMER_RELEASE
#ifndef CUSTOMER_NAME
#error CUSTOMER_RELEASE is defined. You must set CUSTOMER_NAME.
#endif
	// If the user does not want us to check for new versions, just don't do it.
	if (!pvconfig.value("check_new_versions", true).toBool()) {
		return 1;
	}

	QNetworkAccessManager *manager = new QNetworkAccessManager(this);
	QNetworkRequest request;

	connect(manager, SIGNAL(finished(QNetworkReply*)),
		this, SLOT(update_reply_finished_Slot(QNetworkReply*)));

	//request.setUrl(QUrl("http://www.picviz.com/update.html"));
	request.setUrl(QUrl(PVCore::PVVersion::update_url()));
	request.setRawHeader("User-Agent", "Mozilla/5.0 (X11; Linux x86_64; rv:5.0) Gecko/20100101 Firefox/5.0 " CUSTOMER_NAME " PV/" PICVIZ_CURRENT_VERSION_STR);

	manager->get(request);

#endif

	return 0;
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::update_statemachine_label
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::update_statemachine_label(Picviz::PVView_sp view)
{
	statemachine_label->setText(view->state_machine->get_string());
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::SceneMenuEventFilter::eventFilter
 *
 *****************************************************************************/
bool PVInspector::PVMainWindow::SceneMenuEventFilter::eventFilter(QObject* obj, QEvent* event)
{
	if(event->type() == QEvent::Show) {
		bool is_enabled = false;
		Picviz::PVScene* s = _parent->current_scene();
		if (s) {
			uint32_t nb_sources = _parent->current_scene()->get_children<Picviz::PVSource>().size();
			is_enabled = nb_sources >= 2;
		}
		_parent->correlation_scene_Action->setEnabled(is_enabled);
		return true;
	}
	return QObject::eventFilter(obj, event);
}
