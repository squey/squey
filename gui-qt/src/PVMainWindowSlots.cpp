/**
 * \file PVMainWindowSlots.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <pvkernel/core/qobject_helpers.h>

#include <pvkernel/core/PVAxesIndexType.h>
#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/core/PVRecentItemsManager.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/core/PVSerializeArchiveZip.h>
#include <pvkernel/core/PVSerializeArchiveFixError.h>
#include <pvkernel/core/PVVersion.h>
#include <pvkernel/core/PVConfig.h>

#include <picviz/PVAxisComputation.h>
#include <picviz/PVPlotting.h>
#include <picviz/PVMapping.h>

#include <picviz/widgets/editors/PVAxisIndexEditor.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVCallHelper.h>

#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>

#include <pvguiqt/PVAxesCombinationDialog.h>
#include <pvguiqt/PVLayerFilterProcessWidget.h>
#include <pvguiqt/PVImportSourceToProjectDlg.h>
#include <pvguiqt/PVWorkspace.h>
#include <pvguiqt/PVWorkspacesTabWidget.h>
#include <pvguiqt/PVExportSelectionDlg.h>
#include <pvguiqt/PVAboutBoxDialog.h>

#include <PVMainWindow.h>
#include <PVExpandSelDlg.h>
#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <picviz/widgets/PVArgumentListWidgetFactory.h>
#include <PVFormatBuilderWidget.h>
#include <PVExtractorWidget.h>
#include <PVAxisComputationDlg.h>
#include <PVSaveDataTreeDialog.h>

#include <QPainter>
#include <QDockWidget>

/******************************************************************************
 *
 * PVInspector::PVMainWindow::about_Slot()
 *
 *****************************************************************************/

int PVInspector::PVMainWindow::sequence_n = 1;

void PVInspector::PVMainWindow::about_Slot()
{
	PVGuiQt::PVAboutBoxDialog* about_dialog = new PVGuiQt::PVAboutBoxDialog(this);
	about_dialog->exec();
	about_dialog->deleteLater();
}

void PVInspector::PVMainWindow::axes_editor_Slot()
{
	if (!current_view()) {
		return;
	}
}

void PVInspector::PVMainWindow::axes_combination_editor_Slot()
{
	if (!current_view()) {
		return;
	}

	PVGuiQt::PVAxesCombinationDialog* dlg = ((PVGuiQt::PVWorkspaceBase*) _projects_tab_widget->current_workspace())->get_axes_combination_editor(current_view());
	if (dlg->isVisible()) {
		return;
	}

	dlg->reset_used_axes();
	dlg->show();
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::axes_mode_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::axes_mode_Slot()
{
	PVLOG_INFO("%s\n", __FUNCTION__);
	Picviz::PVView* current_lib_view;

	if (!current_view()) {
		return;
	}
	current_lib_view = current_view();

	current_lib_view->state_machine->toggle_axes_mode();

	// if we enter in AXES_MODE we must disable SQUARE_AREA_MODE
	if (current_lib_view->state_machine->is_axes_mode()) {
		/* We turn SQUARE AREA mode OFF */
		current_lib_view->state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_OFF);
		//current_view->update_axes();
		axes_mode_Action->setText(QString("Leave Axes mode"));
	} else {
		axes_mode_Action->setText(QString("Enter Axes mode"));
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::axes_display_edges_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::axes_display_edges_Slot()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::commit_selection_in_current_layer_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::commit_selection_in_current_layer_Slot()
{
	/* We prepare a direct access to the current lib_view */
	Picviz::PVView* current_lib_view;

	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);

	if (_projects_tab_widget->current_workspace() == nullptr) {
		return;
	}
	current_lib_view = current_view();
	commit_selection_in_current_layer(current_lib_view);
}

/******************************************************************************
 *
* PVInspector::PVMainWindow::commit_selection_to_new_layer_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::commit_selection_to_new_layer_Slot()
{
	if (!current_view()) {
		return;
	}

	Picviz::PVView* current_lib_view = current_view();
	commit_selection_to_new_layer(current_lib_view);
}

void PVInspector::PVMainWindow::move_selection_to_new_layer_Slot()
{
	if (!current_view()) {
		return;
	}

	Picviz::PVView* current_lib_view = current_view();
	move_selection_to_new_layer(current_lib_view);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::events_display_unselected_listing_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::events_display_unselected_listing_Slot()
{
	Picviz::PVView* current_lib_view;

	if (!current_view()) {
		return;
	}
	current_lib_view = current_view();

	/* We refresh the listing */
	Picviz::PVView_sp view_sp = current_lib_view->shared_from_this();
	PVHive::call<FUNC(Picviz::PVView::toggle_listing_unselected_visibility)>(view_sp);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::events_display_unselected_GLview_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::events_display_unselected_GLview_Slot()
{
	Picviz::PVView* current_lib_view;
	Picviz::PVStateMachine *state_machine = NULL;

	if (!current_view()) {
		return;
	}
	current_lib_view = current_view();
	state_machine = current_lib_view->state_machine;

	if (_projects_tab_widget->current_workspace() == nullptr) {
		return;
	}

	state_machine->toggle_gl_unselected_visibility();
	/* We refresh the view */
	current_lib_view->process_visibility();
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::events_display_zombies_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::events_display_zombies_Slot()
{
	Picviz::PVView* current_lib_view;
	Picviz::PVStateMachine *state_machine = NULL;

	if (!current_view()) {
		return;
	}
	current_lib_view = current_view();
	state_machine = current_lib_view->state_machine;

	state_machine->toggle_listing_zombie_visibility();
	state_machine->toggle_gl_zombie_visibility();
	/* We set the listing to be the same */
	// state_machine->set_listing_zombie_visibility(state_machine->are_zombie_visible());
	/* We refresh the view */
	current_lib_view->process_visibility();
	/* We refresh the listing */
	// TODO: hive!
	//current_tab->update_pv_listing_model_Slot();

}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::events_display_zombies_listing_Sloupdate_recent_projectst()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::events_display_zombies_listing_Slot()
{
	Picviz::PVView* current_lib_view;

	if (!current_view()) {
		return;
	}
	current_lib_view = current_view();

	Picviz::PVView_sp view_sp = current_lib_view->shared_from_this();
	PVHive::call<FUNC(Picviz::PVView::toggle_listing_zombie_visibility)>(view_sp);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::events_display_zombies_GLview_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::events_display_zombies_GLview_Slot()
{
	Picviz::PVView* current_lib_view;
	Picviz::PVStateMachine *state_machine = NULL;

	if (!current_view()) {
		return;
	}
	current_lib_view = current_view();
	state_machine = current_lib_view->state_machine;

	state_machine->toggle_gl_zombie_visibility();
	/* We refresh the view */
	current_lib_view->process_visibility();
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::events_display_unselected_zombies_parallelview_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::events_display_unselected_zombies_parallelview_Slot()
{
	Picviz::PVView* current_lib_view;

	if (!current_view()) {
		return;
	}
	current_lib_view = current_view();

	/* We refresh the listing */
	Picviz::PVView_sp view_sp = current_lib_view->shared_from_this();
	PVHive::call<FUNC(Picviz::PVView::toggle_view_unselected_zombie_visibility)>(view_sp);
}

void PVInspector::PVMainWindow::expand_selection_on_axis_Slot()
{
	if (!current_view()) {
		return;
	}
	Picviz::PVView* cur_view_p = current_view();
	PVExpandSelDlg* dlg = new PVExpandSelDlg(*cur_view_p, this);
	Picviz::PVView &view = *cur_view_p;
	if (dlg->exec() != QDialog::Accepted) {
		return;
	}

	PVCore::PVAxesIndexType axes = dlg->get_axes();
	PVCore::PVAxesIndexType::const_iterator it;
	QString mode = dlg->get_mode();
	for (it = axes.begin(); it != axes.end(); it++) {
		view.expand_selection_on_axis(*it, mode);
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::export_file_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::export_file_Slot()
{

}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::export_selection_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::export_selection_Slot()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);

	Picviz::PVView* view = current_view();
	Picviz::PVSelection& sel = view->get_real_output_selection();

	PVGuiQt::PVExportSelectionDlg::export_selection(*view, sel);
}


/******************************************************************************
 *
 * PVInspector::PVMainWindow::filter_select_all_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::filter_select_all_Slot()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);

	if (!current_view()) {
		return;
	}
	/* We do all that has to be done in the lib FIRST */
	current_view()->apply_filter_named_select_all();
	current_view()->process_from_eventline();

	/* THEN we can emit the signal */
	emit selection_changed_Signal();

}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::filter_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::filter_Slot(void)
{
	if (current_view()) {
		QObject *s = sender();
		Picviz::PVView* lib_view = current_view();
		QString filter_name = s->objectName();

		Picviz::PVLayerFilter::p_type filter_org = LIB_CLASS(Picviz::PVLayerFilter)::get().get_class_by_name(filter_name);
		Picviz::PVLayerFilter::p_type fclone = filter_org->clone<Picviz::PVLayerFilter>();
		PVCore::PVArgumentList &args = lib_view->get_last_args_filter(filter_name);
		PVGuiQt::PVLayerFilterProcessWidget* filter_widget = new PVGuiQt::PVLayerFilterProcessWidget(current_view(), args, fclone);
		filter_widget->show();
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::filter_reprocess_last_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::filter_reprocess_last_Slot()
{
	if (current_view()) {
		Picviz::PVView* lib_view = current_view();
		if (!lib_view->is_last_filter_used_valid()) {
			return;
		}
		QString const& filter_name = lib_view->get_last_used_filter();
		Picviz::PVLayerFilter::p_type filter_org = LIB_CLASS(Picviz::PVLayerFilter)::get().get_class_by_name(filter_name);
		Picviz::PVLayerFilter::p_type fclone = filter_org->clone<Picviz::PVLayerFilter>();
		PVCore::PVArgumentList &args = lib_view->get_last_args_filter(filter_name);
		PVGuiQt::PVLayerFilterProcessWidget* filter_widget = new PVGuiQt::PVLayerFilterProcessWidget(current_view(), args, fclone);
		filter_widget->show();
		filter_widget->preview_Slot();
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::extractor_file_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::extractor_file_Slot()
{
	if (!current_view()) {
		//TODO: this should not happen because the menu item should be disabled... !
		return;
	}
	
	// For now, shows a modal dialog!
	PVExtractorWidget* ext = new PVExtractorWidget(*current_view()->get_parent<Picviz::PVSource>(), _projects_tab_widget, this);
	ext->exec();
	ext->deleteLater();
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::map_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::map_Slot()
{

	QDialog *dialog = new QDialog;
	QHBoxLayout *layout = new QHBoxLayout;


	//	GKMapView *mapView = new GKMapView;

 //    mapView->resize(580, 410);
 //  //  mapView->setMapType(GKMapTypeSatellite);
 // //   mapView->locationFromAddress("San Mateo");
 //   // mapView->addressFromLocation(32.718834, -117.164);
 //  mapView->addMarkerWithWindow("Attacks seen", "We have an intrusion attemp in France!", "Paris, France");
 // //   mapView.addInfoWindow("Hello from <b>San Mateo</b>", "San Mateo");
 //   // mapView.addInfoWindow("Hello from <b>San Diego</b>", 32.718834, -117.164);
 //   mapView->setLocation("France");
 //    mapView->show();

 //    layout->addWidget(mapView);
//	main_layout->addWidget(mapView, 0, 0);

	dialog->setLayout(layout);

	dialog->setWindowTitle("Map Widget");

	dialog->show();

}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::project_new_Slot
 *
 *****************************************************************************/
Picviz::PVScene_p PVInspector::PVMainWindow::project_new_Slot()
{
	QString scene_name = tr("Data collection %1").arg(sequence_n++);
	PVCore::PVDataTreeAutoShared<Picviz::PVScene> scene_p = PVCore::PVDataTreeAutoShared<Picviz::PVScene>(get_root_sp(), scene_name);
	_projects_tab_widget->add_project(scene_p);

	return scene_p;
}

void PVInspector::PVMainWindow::load_source_from_description_Slot(PVRush::PVSourceDescription src_desc)
{
	Picviz::PVScene_sp scene_p;

	PVRush::PVFormat format = src_desc.get_format();
	if ((format.exists() == false) || (QFileInfo(format.get_full_path()).isReadable() == false)) {
		QMessageBox::warning(this,
		                     tr("Format \"%1\" can not be read").arg(format.get_format_name()),
		                     tr("Check that the file \"%1\" exists and is readable").arg(format.get_full_path()));
		return;
	}

	QList<Picviz::PVScene_p> scenes = get_root().get_children();

	bool new_scene = false;
	if (scenes.size() == 0) {
		// No loaded project: create a new one and load the source
		scene_p = project_new_Slot();
		new_scene = true;
	}
	else if (scenes.size() == 1) {
		// Only one project loaded: use it to load the source
		scene_p = scenes.at(0)->shared_from_this();
		Picviz::PVRoot_sp root_sp = get_root().shared_from_this();
		PVHive::call<FUNC(Picviz::PVRoot::select_scene)>(root_sp, *scene_p.get());
	}
	else {
		// More than one project loaded: ask the user the project he wants to use to load the source
		PVGuiQt::PVImportSourceToProjectDlg* dlg = new PVGuiQt::PVImportSourceToProjectDlg(get_root(), get_root().current_scene(), this);
		if (dlg->exec() != QDialog::Accepted) {
			return;
		}

		Picviz::PVRoot_sp root_sp = get_root().shared_from_this();
		PVHive::call<FUNC(Picviz::PVRoot::select_scene)>(root_sp, *((Picviz::PVScene*) dlg->get_selected_scene()));
		scene_p = current_scene()->shared_from_this();
		dlg->deleteLater();
	}

	Picviz::PVSource_sp src_p;
	try {
		 src_p = PVHive::call<FUNC(Picviz::PVScene::add_source_from_description)>(scene_p, src_desc);
	}
	catch (PVRush::PVInputException const& e) {
		if (new_scene) {
			_projects_tab_widget->remove_project(_projects_tab_widget->get_workspace_tab_widget_from_scene(scene_p.get()));
		}
		QMessageBox::critical(this, tr("Fatal error while loading source..."), tr("Fatal error while loading source: %1").arg(e.what().c_str()));
		return;

	}

	if (!load_source(src_p.get())) {
		remove_source(src_p.get());
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::project_load_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::project_load_Slot()
{
	/*
#ifdef CUSTOMER_CAPABILITY_SAVE
	_load_solution_dlg.setFileMode(QFileDialog::ExistingFile);
	_load_solution_dlg.setAcceptMode(QFileDialog::AcceptOpen);
	if (_load_solution_dlg.exec() != QDialog::Accepted) {
		return;
	}
	QString file = _load_solution_dlg.selectedFiles().at(0);

	load_project(file);
#endif
	*/
}

void PVInspector::PVMainWindow::solution_new_Slot()
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	PVMainWindow* new_mw = new PVMainWindow();
	new_mw->move(x() + 40, y() + 40);
	new_mw->show();
	new_mw->set_window_title_with_filename();
#endif
}

void PVInspector::PVMainWindow::solution_load_Slot()
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	_load_solution_dlg.setFileMode(QFileDialog::ExistingFile);
	_load_solution_dlg.setAcceptMode(QFileDialog::AcceptOpen);
	if (_load_solution_dlg.exec() != QDialog::Accepted) {
		return;
	}
	QString file = _load_solution_dlg.selectedFiles().at(0);
	load_solution_and_create_mw(file);
#endif
}

void PVInspector::PVMainWindow::load_solution_and_create_mw(QString const& file)
{
	PVMainWindow* existing = find_main_window(file);
	if (existing) {
		existing->show();
		existing->raise();
		existing->activateWindow();
		return;
	}
	if (is_solution_untitled() && get_root().is_empty() && !isWindowModified()) {
		load_solution(file);
	}
	else {
		PVMainWindow* other = new PVMainWindow();
		other->move(x() + 40, y() + 40); 
		other->show();
		if (!load_solution(file)) {
			other->deleteLater();
			return;
		}
	}
}

void PVInspector::PVMainWindow::solution_save_Slot()
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	if (is_solution_untitled()) {
		solution_saveas_Slot();
	}
	else {
		PVCore::PVSerializeArchiveOptions_p options(get_root().get_default_serialize_options());
		save_solution(get_solution_path(), options);
	}
#endif
}

void PVInspector::PVMainWindow::solution_saveas_Slot()
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	if (get_root().is_empty()) {
		return;
	}

	PVCore::PVSerializeArchiveOptions_p options(get_root().get_default_serialize_options());
	PVSaveDataTreeDialog* dlg = new PVSaveDataTreeDialog(options, PICVIZ_ROOT_ARCHIVE_EXT, PICVIZ_ROOT_ARCHIVE_FILTER, this);
	if (!_current_save_root_folder.isEmpty()) {
		dlg->setDirectory(_current_save_root_folder);
	}    
	dlg->selectFile(get_solution_path());
	if (dlg->exec() == QDialog::Accepted) {
		QString file = dlg->selectedFiles().at(0);
		save_solution(file, options);
	}    
	_current_save_root_folder = dlg->directory().absolutePath();
	dlg->deleteLater();
#endif
}

bool PVInspector::PVMainWindow::maybe_save_solution()
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	if (isWindowModified()) {
		QMessageBox::StandardButton ret;
		QString solution_name = QFileInfo(windowFilePath()).fileName();
		ret = QMessageBox::warning(this, tr("%1").arg(solution_name),
				tr("The solution \"%1\"has been modified.\n"
					"Do you want to save your changes?").arg(solution_name),
				QMessageBox::Save | QMessageBox::Discard
				| QMessageBox::Cancel);
		if (ret == QMessageBox::Save) {
			solution_save_Slot();
			return true;
		}
		if (ret == QMessageBox::Discard) {
			return true;
		}
		else if (ret == QMessageBox::Cancel) {
			return false;
		}
	}
	return true;
#else
	return false;
#endif
}

bool PVInspector::PVMainWindow::load_solution(QString const& file)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	setWindowModified(false);

	PVCore::PVSerializeArchive_p ar;
	PVCore::PVSerializeArchiveError read_exception = PVCore::PVSerializeArchiveError(QString());
	PVCore::PVProgressBox* pbox_solution = new PVCore::PVProgressBox("Loading investigation...", this);
	pbox_solution->set_enable_cancel(true);
	bool ret = PVCore::PVProgressBox::progress([&] {
			try {
				ar.reset(new PVCore::PVSerializeArchiveZip(file, PVCore::PVSerializeArchive::read, PICVIZ_ARCHIVES_VERSION));
			} catch (const PVCore::PVSerializeArchiveError& e) {
				read_exception = e;
			}
		}, pbox_solution);
	if (!ret) {
		return false;
	}

	if (!read_exception.what().isEmpty()) {
		QMessageBox* box = new QMessageBox(QMessageBox::Critical, tr("Fatal error while loading solution..."), tr("Fatal error while loading solution %1:\n%2").arg(file).arg(read_exception.what()), QMessageBox::Ok, this);
		box->exec();
		return false;
	}    

	bool solution_has_been_fixed = false;
	while (true) {
		QString err_msg;
		try {
			get_root().load_from_archive(ar);
		}    
		catch (PVCore::PVSerializeArchiveError& e) { 
			err_msg = tr("Error while loading solution %1:\n%2").arg(file).arg(e.what());
		}    
		catch (PVRush::PVInputException const& e)
		{    
			err_msg = tr("Error while loading solution %1:\n%2").arg(file).arg(QString::fromStdString(e.what()));
		}    
		catch (...)
		{    
			err_msg = tr("Fatal error while loading solution %1:\n unhandled error(s).").arg(file);
		}    
		if (!err_msg.isEmpty()) {
			QMessageBox* box = new QMessageBox(QMessageBox::Critical, tr("Fatal error while loading solution..."), err_msg, QMessageBox::Ok, this);
			box->exec();
			return false;
		}
		if (ar->has_repairable_errors()) {
			if (fix_project_errors(ar)) {
				solution_has_been_fixed = true;
				reset_root();
				continue;
			}    
			else {
				if (!err_msg.isEmpty()) {
					QMessageBox* box = new QMessageBox(QMessageBox::Critical, tr("Error while loading solution..."), err_msg, QMessageBox::Ok, this);
					box->exec();
				}    
				reset_root();
				return false;
			}
		}
		break;
	}

	if (!load_root()) {
		PVLOG_ERROR("(PVMainWindow::solution_load) error while processing the solution...\n");
		reset_root();
		return false;
	}

	_root->set_path(file);

#ifdef ENABLE_CORRELATION
	correlation_Menu->load_correlations();
#endif

	set_window_title_with_filename();
	if (solution_has_been_fixed) {
		setWindowModified(true);
	}

	PVHive::call<FUNC(PVCore::PVRecentItemsManager::add)>(PVCore::PVRecentItemsManager::get(), file, PVCore::PVRecentItemsManager::Category::PROJECTS);

	flag_investigation_as_cached(file);

	return true;
#endif

	return false;
}

void PVInspector::PVMainWindow::save_solution(QString const& file, PVCore::PVSerializeArchiveOptions_p const& options)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	try {
		PVCore::PVProgressBox* pbox_solution = new PVCore::PVProgressBox("Saving investigation...", this);
		pbox_solution->set_enable_cancel(true);
		bool ret = PVCore::PVProgressBox::progress([&] {get_root().save_to_file(file, options);}, pbox_solution);
		if (!ret) {
			return;
		}
	}
	catch (PVCore::PVSerializeArchiveError const& e) {
		QMessageBox* box = new QMessageBox(QMessageBox::Critical, tr("Error while saving solution..."), tr("Error while saving solution %1:\n%2").arg(file).arg(e.what()), QMessageBox::Ok, this);
		box->exec();
	}

	PVHive::call<FUNC(PVCore::PVRecentItemsManager::add)>(PVCore::PVRecentItemsManager::get(), file, PVCore::PVRecentItemsManager::Category::PROJECTS);

	flag_investigation_as_cached(file);

	set_window_title_with_filename();
#endif
}

void PVInspector::PVMainWindow::flag_investigation_as_cached(const QString& investigation)
{
	QStringList nraws;
	for (Picviz::PVSource_p& source : get_root().get_children<Picviz::PVSource>()) {
		nraws << QString(source->get_rushnraw().get_nraw_folder().c_str());
	}
	PVRush::PVNrawCacheManager::get().add_investigation(investigation, nraws);
}

void PVInspector::PVMainWindow::set_window_title_with_filename()
{
	static int sequenceNumber = 1;

	QString file;
	if (is_solution_untitled()) {
		file = tr("new-solution%1." PICVIZ_ROOT_ARCHIVE_EXT).arg(sequenceNumber++);
	} else {
		file = QFileInfo(get_solution_path()).canonicalFilePath();
	}

	setWindowModified(false);
	setWindowFilePath(file);
}


void PVInspector::PVMainWindow::create_new_window_for_workspace(QWidget* widget_workspace)
{
	PVMainWindow* other = new PVMainWindow();
	other->move(QCursor::pos());
	other->resize(size());
	other->show();

	//other->_workspaces_tab_widget->setVisible(true);

	PVGuiQt::PVSourceWorkspace* workspace = dynamic_cast<PVGuiQt::PVSourceWorkspace*>(widget_workspace);
	if (workspace) {
		_projects_tab_widget->remove_workspace(workspace, false);
		other->_projects_tab_widget->add_workspace((PVGuiQt::PVSourceWorkspace*) workspace);
	}
}

bool PVInspector::PVMainWindow::fix_project_errors(PVCore::PVSerializeArchive_p ar)
{
	// Fix errors due to invalid file paths
	PVCore::PVSerializeArchive::list_errors_t errs_file = ar->get_repairable_errors_of_type<PVCore::PVSerializeArchiveErrorFileNotReadable>();
	// TODO: a nice widget were file paths can be modified by batch (for instance modify all the files' directory in one action)
	foreach(PVCore::PVSerializeArchiveFixError_p err, errs_file) {
		QString const& old_path(err->exception_as<PVCore::PVSerializeArchiveErrorFileNotReadable>()->get_path());
		QMessageBox* box = new QMessageBox(QMessageBox::Warning, tr("Error while loading project..."), tr("File '%1' cannot be found or isn't readable by the process. Please select its new path.").arg(old_path), QMessageBox::Ok, this);
		box->exec();
		QString new_file = QFileDialog::getOpenFileName(this, tr("Select new file path..."), old_path);
		if (new_file.isEmpty()) {
			return false;
		}
		PVCore::PVSerializeArchiveFixAttribute* fix_a = (PVCore::PVSerializeArchiveFixAttribute*) err.get();
		fix_a->fix(new_file);
	}

	// Return true if and only if all the errors have been fixed.
	return !ar->has_repairable_errors();
}

bool PVInspector::PVMainWindow::load_project(QString const& /*file*/)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	/*if (!maybe_save_project()) {
		return false;
	}
	set_project_modified(false);*/

	/*close_scene();*/


	/*
	Picviz::PVScene* scene = get_root().get_scene_from_path(file);

	if (scene) {
		Picviz::PVRoot_sp root_sp = get_root().shared_from_this();
		PVHive::call<FUNC(Picviz::PVRoot::select_scene)>(root_sp, *scene);
		return false;
	}

	PVCore::PVDataTreeAutoShared<Picviz::PVScene> scene_p = PVCore::PVDataTreeAutoShared<Picviz::PVScene>(get_root_sp(), file);

	PVCore::PVSerializeArchive_p ar;
	try {
		ar.reset(new PVCore::PVSerializeArchiveZip(file, PVCore::PVSerializeArchive::read, PICVIZ_ARCHIVES_VERSION));
	}
	catch (PVCore::PVSerializeArchiveError& e) {
		QMessageBox* box = new QMessageBox(QMessageBox::Critical, tr("Error while loading project..."), tr("Error while loading project %1:\n%2").arg(file).arg(e.what()), QMessageBox::Ok, this);
		box->exec();
		return false;
	}

	//bool project_has_been_fixed = false;
	while (true) {
		QString err_msg;
		try {
			scene_p->load_from_archive(ar);
		}
		catch (PVCore::PVSerializeArchiveError& e) {
			err_msg = tr("Error while loading project %1:\n%2").arg(file).arg(e.what());
		}
		catch (PVRush::PVInputException const& e)
		{
			err_msg = tr("Error while loading project %1:\n%2").arg(file).arg(QString::fromStdString(e.what()));
		}
		catch (...)
		{
			err_msg = tr("Error while loading project %1:\n unhandled error.").arg(file);
			QMessageBox* box = new QMessageBox(QMessageBox::Critical, tr("Error while loading project..."), err_msg, QMessageBox::Ok, this);
			box->exec();
			return false;
		}
		if (ar->has_repairable_errors()) {
			if (fix_project_errors(ar)) {
				_root.reset(new Picviz::PVRoot());
				continue;
			}
			else {
				if (!err_msg.isEmpty()) {
					QMessageBox* box = new QMessageBox(QMessageBox::Critical, tr("Error while loading project..."), err_msg, QMessageBox::Ok, this);
					box->exec();
				}
				//close_scene();
				//_scene.reset();
				return false;
			}
		}
		break;
	}

	if (!load_scene(scene_p.get())) {
		PVLOG_ERROR("(PVMainWindow::project_load_Slot) error while processing the scene...\n");
		//close_scene();
		//_scene.reset();
		return false;
	}

	//_projects_tab_widget->add_project(scene_p);

	show_start_page(false);
	//_workspaces_tab_widget->setVisible(true);

	PVHive::call<FUNC(PVCore::PVRecentItemsManager::add)>(PVCore::PVRecentItemsManager::get(), file, PVCore::PVRecentItemsManager::Category::PROJECTS);
	*/
#endif

	return true;
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::project_save_Slot
 *
 *****************************************************************************/
bool PVInspector::PVMainWindow::project_save_Slot()
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	if (is_project_untitled()) {
		return project_saveas_Slot();
	}
	else {
		PVCore::PVSerializeArchiveOptions_p options(current_scene()->get_default_serialize_options());
		return save_project(_cur_project_file, options);
	}
#endif
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::project_saveas_Slot
 *
 *****************************************************************************/
bool PVInspector::PVMainWindow::project_saveas_Slot()
{
	bool ret = false;
#ifdef CUSTOMER_CAPABILITY_SAVE
	if (current_scene()) {
		PVCore::PVSerializeArchiveOptions_p options(current_scene()->get_default_serialize_options());
		PVSaveDataTreeDialog* dlg = new PVSaveDataTreeDialog(options, PICVIZ_SCENE_ARCHIVE_EXT, PICVIZ_SCENE_ARCHIVE_FILTER, this);
		/*if (!_current_save_project_folder.isEmpty()) {
			dlg->setDirectory(_current_save_project_folder);
		}*/
		dlg->selectFile(current_scene()->get_path());
		if (dlg->exec() == QDialog::Accepted) {
			QString file = dlg->selectedFiles().at(0);
			ret = save_project(file, options);
		}
		//_current_save_project_folder = dlg->directory().absolutePath();
		dlg->deleteLater();
	}
#endif
	return ret;
}

bool PVInspector::PVMainWindow::save_project(QString const& file, PVCore::PVSerializeArchiveOptions_p options)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	try {
		Picviz::PVScene_p scene_p = current_scene()->shared_from_this();
		PVHive::call<FUNC(Picviz::PVScene::save_to_file)>(scene_p, file, options, false);
	}
	catch (PVCore::PVSerializeArchiveError& e) {
		QMessageBox* box = new QMessageBox(QMessageBox::Critical, tr("Error while saving project..."), tr("Error while saving project %1:\n%2").arg(file).arg(e.what()), QMessageBox::Ok, this);
		box->exec();
		return false;
	}

	PVHive::call<FUNC(PVCore::PVRecentItemsManager::add)>(PVCore::PVRecentItemsManager::get(), file, PVCore::PVRecentItemsManager::Category::PROJECTS);

	return true;
#else
	return false;
#endif
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::quit_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::quit_Slot()
{
	close();
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::refresh_current_view_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::refresh_current_view_Slot()
{
	// FIXME: this function should probably just die. current_tab->refresh_view_Slot();
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::select_scene_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::select_scene_Slot()
{
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::selection_inverse_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::selection_all_Slot()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);
	if (!current_view()) {
		return;
	}

	Picviz::PVView_sp lib_view(current_view()->shared_from_this());
	if (lib_view) {
		lib_view->select_all_nonzb_lines();
		PVHive::PVCallHelper::call<FUNC(Picviz::PVView::process_real_output_selection)>(lib_view);
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::selection_inverse_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::selection_none_Slot()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);
	if (!current_view()) {
		return;
	}

	Picviz::PVView_sp lib_view(current_view()->shared_from_this());
	if (lib_view) {
		lib_view->select_no_line();
		PVHive::PVCallHelper::call<FUNC(Picviz::PVView::process_real_output_selection)>(lib_view);
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::selection_inverse_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::selection_inverse_Slot()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);
	if (!current_view()) {
		return;
	}

	Picviz::PVView_sp lib_view(current_view()->shared_from_this());
	if (lib_view) {
		lib_view->select_inv_lines();
		PVHive::PVCallHelper::call<FUNC(Picviz::PVView::process_real_output_selection)>(lib_view);
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::set_color_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::set_color_Slot()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);

	/* CODE */
	if (!current_view())
		return;
	set_color(current_view());
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::textedit_text_changed_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::textedit_text_changed_Slot()
{
#if 0 // FIXME
	/* VARIABLES */
	QObject *s = sender();
	QLineEdit *lineedit = reinterpret_cast<QLineEdit *>(s);
	char *text = strdup(lineedit->text().toUtf8().data());

	picviz_arguments_t *args;
	picviz_argument_item_t item;
	char *sender_name;

	if (strcmp(text, "")) {
		sender_name = strdup(s->objectName().toUtf8().data());

		args = filter->get_arguments_func();

		item = picviz_arguments_get_item_from_name(args, sender_name);
		picviz_arguments_item_set_string(item, text);
		picviz_arguments_set_item_from_name(args, item.name, item);

		picviz_arguments_debug(args);

		current_view()->apply_filter_from_name(last_sendername, args);
		current_view()->process_from_eventline();

		free(sender_name);

		/* THEN we can emit the signal */
		emit filter_applied_Signal();
	}

	free(text);
#endif
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::update_reply_finished_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::update_reply_finished_Slot(QNetworkReply *reply)
{
	if (reply->error() != QNetworkReply::NoError) {
		// There was an error retrieving the current version.
		// Maybe picviz has no internet access !
		PVLOG_DEBUG("(PVMainWindow::update_reply_finished_Slot) network error\n");
		set_version_informations();
		return;
	}

	QByteArray data = reply->readAll();
	version_t current_v, last_v;
	if (!PVCore::PVVersion::from_network_reply(data, current_v, last_v)) {
		PVLOG_DEBUG("(PVMainWindow::update_reply_finished_Slot) invalid server reply:\n%s\n", qPrintable(QString::fromUtf8(data.constData(), data.size())));
		return;
	}

	if (PICVIZ_MAJOR_VERSION(current_v) != PICVIZ_CURRENT_VERSION_MAJOR ||
		PICVIZ_MINOR_VERSION(current_v) != PICVIZ_CURRENT_VERSION_MINOR ||
		last_v < PICVIZ_CURRENT_VERSION) {
		// Invalid answer from the server
		PVLOG_DEBUG("(PVMainWindow::update_reply_finished_Slot) invalid server reply: version mismatch:\ncurrent version: %s / last current major/minor version: %s\nlast available version: %s.",
				PICVIZ_CURRENT_VERSION_STR, qPrintable(PVCore::PVVersion::to_str(current_v)), qPrintable(PVCore::PVVersion::to_str(last_v)));
		return;
	}

	if (current_v == _last_known_cur_release && last_v == _last_known_maj_release) {
		// We already informed the user once.
		// Display version informations
		set_version_informations();
		return;
	}

	_last_known_cur_release = current_v;
	_last_known_maj_release = last_v;

	// Display version informations
	set_version_informations();

	// Update PVCONFIG settings
	QSettings &pvconfig = PVCore::PVConfig::get().config();

	pvconfig.setValue(PVCONFIG_LAST_KNOWN_CUR_RELEASE, current_v);
	pvconfig.setValue(PVCONFIG_LAST_KNOWN_MAJ_RELEASE, last_v);

	QString desc = tr("Your current version is %1.\n").arg(PICVIZ_CURRENT_VERSION_STR);
	bool show_msg = false;
	if (current_v > PICVIZ_CURRENT_VERSION) {
		// A patch is available
		desc += tr("A new version (%1) is available for free for the %2.%3 branch.").arg(PVCore::PVVersion::to_str(current_v)).arg(PICVIZ_CURRENT_VERSION_MAJOR).arg(PICVIZ_CURRENT_VERSION_MINOR);
		desc += "\n";
		show_msg = true;
	}
	if (last_v != current_v && last_v > PICVIZ_CURRENT_VERSION) {
		// A new major release is available
		desc += tr("A new major release (%1) is available.").arg(PVCore::PVVersion::to_str(last_v));
		show_msg = true;
	}

	if (show_msg) {
		PVLOG_INFO(qPrintable(desc));
		QMessageBox msgBox(QMessageBox::Information, tr("New version available"), tr("A new version is available.\n\n") + desc, QMessageBox::Ok, this);
		msgBox.exec();
	}

}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::view_new_scatter_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::view_new_scatter_Slot()
{
	PVLOG_INFO("PVInspector::PVMainWindow::%s\n", __FUNCTION__);
}

/******************************************************************************
 * PVInspector::PVMainWindow::get_screenshot_widget
 *****************************************************************************/

void PVInspector::PVMainWindow::get_screenshot_widget()
{
	/**
	 * to do the screenshot of widgets, we search for the more relevant
	 * widget in the widget hierarchy that is under the mouse cursor.
	 *
	 * the screenshot may be on:
	 * - a dialog (the statistical views)
	 * - any QDockWidget (graphical views, listing, layerstack
	 * - the current workspace
	 * - the whole main window
	 *
	 * Note: As the family tree is fully traversed for each wanted class,
	 * the tests order is really important.
	 *
	 * If the class list to check for needs to be more complicated, there
	 * may be required to rewrite the whole algorithm to pass the whole
	 * class list as template parameter to do the check in one call.
	 */
	int x = QCursor::pos().x();
	int y = QCursor::pos().y();
	QWidget* w = QApplication::widgetAt(x, y);
	QWidget* p;
	QString name;

	p = PVCore::get_qobject_hierarchy_of_type<PVFormatBuilderWidget>(w);
	if (p == nullptr) {
		p = PVCore::get_qobject_hierarchy_of_type<QDialog>(w);
		if (p == nullptr) {
			p = PVCore::get_qobject_hierarchy_of_type<QDockWidget>(w);
			if (p == nullptr) {
				p = PVCore::get_qobject_hierarchy_of_type<PVGuiQt::PVWorkspacesTabWidgetBase>(w);
				if (p == nullptr) {
					p = PVCore::get_qobject_hierarchy_of_type<PVMainWindow>(w);
				}
			}
		}
	} else {
		name = "format-builder";
	}

	if (p == nullptr) {
		return;
	}

	if (name.isEmpty()) {
		if (_projects_tab_widget->current_workspace_tab_widget() != nullptr) {
			/* if there is a workspace_tab_widget, we are on the workspaces
			 * page or on a data collection page
			 */
			int current_tab_index = _projects_tab_widget->current_workspace_tab_widget()->currentIndex();
			name = QFileInfo(_projects_tab_widget->current_workspace_tab_widget()->tabText(current_tab_index)).baseName();
		} else {
			/* if there is no workspace_tab_widget, it means we are on the
			 * start screen
			 */
			name = "startscreen";
		}
	}


	QPixmap pixmap = QPixmap::grabWidget(p);

	save_screenshot(pixmap, "Save view capture", name);
}

/******************************************************************************
 * PVInspector::PVMainWindow::get_screenshot_window
 *****************************************************************************/

void PVInspector::PVMainWindow::get_screenshot_window()
{
	QPixmap pixmap = QPixmap::grabWindow(winId());

	save_screenshot(pixmap,
	                "Save window capture", "application");
}

/******************************************************************************
 * PVInspector::PVMainWindow::get_screenshot_desktop
 *****************************************************************************/

void PVInspector::PVMainWindow::get_screenshot_desktop()
{
	QPixmap pixmap = QPixmap::grabWindow(QApplication::desktop()->winId());

	save_screenshot(pixmap,
	                "Save desktop capture", "desktop");
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::whats_this_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::whats_this_Slot()
{
	QWhatsThis::enterWhatsThisMode();
}



/******************************************************************************
 *
 * PVInspector::PVMainWindow::new_format_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::new_format_Slot() {
    PVFormatBuilderWidget *editorWidget = new PVFormatBuilderWidget(this);
    editorWidget->show();
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::cur_format_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::cur_format_Slot()
{
	Picviz::PVSource* cur_src = nullptr;
	if (current_view()) {
		cur_src = current_view()->get_parent<Picviz::PVSource>();
	}
	if (!cur_src) {
		return;
	}
	PVRush::PVFormat const& format = cur_src->get_format();
	if (format.get_full_path().isEmpty()) {
		return;
	}

    PVFormatBuilderWidget *editorWidget = new PVFormatBuilderWidget(_projects_tab_widget->current_workspace());
	editorWidget->openFormat(format.get_full_path());
    editorWidget->show();
}

void PVInspector::PVMainWindow::edit_format_Slot(const QString& format)
{
    PVFormatBuilderWidget *editorWidget = new PVFormatBuilderWidget(_projects_tab_widget->current_workspace());
	editorWidget->openFormat(format);
    editorWidget->show();
}

void PVInspector::PVMainWindow::open_format_Slot()
{
    PVFormatBuilderWidget *editorWidget = new PVFormatBuilderWidget(this);
    QString url = editorWidget->slotOpen();

    if (!url.isEmpty()) {
        editorWidget->show();
        PVHive::call<FUNC(PVCore::PVRecentItemsManager::add)>(PVCore::PVRecentItemsManager::get(), url, PVCore::PVRecentItemsManager::Category::EDITED_FORMATS);
    }
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::enable_menu_filter_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::enable_menu_filter_Slot(bool f){
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);
	filter_Menu->setEnabled(f);
}

void PVInspector::PVMainWindow::edit_format_Slot(QString const& path, QWidget* parent)
{
    PVFormatBuilderWidget *editorWidget = new PVFormatBuilderWidget(parent);
    editorWidget->show();
	editorWidget->openFormat(path);
}

void PVInspector::PVMainWindow::edit_format_Slot(QDomDocument& doc, QWidget* parent)
{
    PVFormatBuilderWidget *editorWidget = new PVFormatBuilderWidget(parent);
    editorWidget->show();
	editorWidget->openFormat(doc);
}

void PVInspector::PVMainWindow::axes_new_Slot()
{
	if (!current_view()) {
		return;
	}
	
	Picviz::PVView* view = current_view();
	/*
	std::vector<PVCore::PVUnicodeString> vec_str;
	PVRow nrows = view->get_rushnraw_parent().get_number_rows();
	vec_str.reserve(nrows);

	QString* tmp = new QString("test");
	const PVCore::PVUnicodeString::utf_char* buf = (const PVCore::PVUnicodeString::utf_char*) tmp->unicode();
	for (PVRow i = 0; i < nrows; i++) {
		vec_str.push_back(PVCore::PVUnicodeString(buf, tmp->size()));
	}
	*/

	/*
	QDialog* txt_dlg = new QDialog(this);
	QTextEdit* code_edit = new QTextEdit();
	QDialogButtonBox* btns = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	QVBoxLayout* layout_dlg = new QVBoxLayout();
	layout_dlg->addWidget(code_edit);
	layout_dlg->addWidget(btns);
	connect(btns, SIGNAL(accepted()), txt_dlg, SLOT(accept()));
	connect(btns, SIGNAL(rejected()), txt_dlg, SLOT(reject()));
	txt_dlg->setLayout(layout_dlg);
	if (txt_dlg->exec() != QDialog::Accepted) {
		return;
	}

	Picviz::PVAxisComputation::p_type ac_lib = LIB_CLASS(Picviz::PVAxisComputation)::get().get_class_by_name("python");
	Picviz::PVAxisComputation::p_type ac_clone = ac_lib->clone<Picviz::PVAxisComputation>();

	PVCore::PVArgumentList args;
	args["script"] = code_edit->toPlainText();
	ac_clone->set_args(args);
	*/

	PVAxisComputationDlg* dlg = new PVAxisComputationDlg(*view, this);
	if (dlg->exec() != QDialog::Accepted) {
		return;
	}

	Picviz::PVAxisComputation_p ac_plugin = dlg->get_plugin();

	Picviz::PVAxis axis;
	axis.set_type("enum");
	axis.set_mapping("default");
	axis.set_plotting("default");
	axis.set_name("New axis test");

	view->get_parent<Picviz::PVSource>()->add_column(ac_plugin->f(), axis);
}

void PVInspector::PVMainWindow::selection_set_from_current_layer_Slot()
{
	if (current_view()) {
		Picviz::PVView_sp view(current_view()->shared_from_this());
		set_selection_from_layer(view, view->get_current_layer());
	}
}

void PVInspector::PVMainWindow::selection_set_from_layer_Slot()
{
	if (current_view()) {
		Picviz::PVView_sp view(current_view()->shared_from_this());

		PVCore::PVArgumentList args;
		args[PVCore::PVArgumentKey("sel-layer", tr("Choose a layer"))].setValue<Picviz::PVLayer*>(&view->get_current_layer());
		bool ret = PVWidgets::PVArgumentListWidget::modify_arguments_dlg(PVWidgets::PVArgumentListWidgetFactory::create_layer_widget_factory(*view), args, this);
		if (ret) {
			Picviz::PVLayer* layer = args["sel-layer"].value<Picviz::PVLayer*>();
			set_selection_from_layer(view, *layer);
		}
	}
}

void PVInspector::PVMainWindow::view_display_inv_elts_Slot()
{
	if (current_view()) {
		display_inv_elts();
	}
}

void PVInspector::PVMainWindow::show_correlation_Slot()
{
	/*if (!_ad2g_mw) {
		_ad2g_mw = new QDialog(this);
		_ad2g_mw->setWindowTitle(tr("Correlations"));
		PVWidgets::PVAD2GWidget* ad2g_w = new PVWidgets::PVAD2GWidget(current_scene()->get_ad2g_view_p());
		QVBoxLayout* l = new QVBoxLayout();
		l->addWidget(ad2g_w);
		_ad2g_mw->setLayout(l);
	}
	else {
		QWidget* ad2g_mw_c = _ad2g_mw->layout()->itemAt(0)->widget();
		PVWidgets::PVAD2GWidget* ad2g_w;
		ad2g_w = dynamic_cast<PVWidgets::PVAD2GWidget*>(ad2g_mw_c);
		assert(ad2g_w);
		ad2g_w->update_list_views();
		ad2g_w->update_list_edges();
	}
	_ad2g_mw->exec();*/
}

void PVInspector::PVMainWindow::layer_export_Slot()
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	if (current_view() == nullptr) {
		return;
	}

	QFileDialog fd;
	QString file = fd.getSaveFileName(this, "Export current layer...",
	                                  fd.directory().absolutePath(),
	                                  PICVIZ_LAYER_ARCHIVE_FILTER ";;" ALL_FILES_FILTER);
	if(!file.isEmpty()) {
		current_view()->get_current_layer().save_to_file(file);
	}
#endif
}

void PVInspector::PVMainWindow::layer_import_Slot()
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	if (current_view() == nullptr) {
		return;
	}

	QFileDialog fd;
	QString file = fd.getOpenFileName(this, "Import a layer...",
	                                  fd.directory().absolutePath(),
	                                  PICVIZ_LAYER_ARCHIVE_FILTER ";;" ALL_FILES_FILTER);
	if(file.isEmpty()) {
		return;
	}

	Picviz::PVView_sp lib_view(current_view()->shared_from_this());
	if (lib_view) {
		lib_view->get_current_layer().reset_to_default_color();
		PVHive::PVCallHelper::call<FUNC(Picviz::PVView::add_new_layer_from_file)>(lib_view, file);
		PVHive::PVCallHelper::call<FUNC(Picviz::PVView::process_from_layer_stack)>(lib_view);
	}
#endif
}

void PVInspector::PVMainWindow::layer_save_ls_Slot()
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	if (current_view() == nullptr) {
		return;
	}

	QFileDialog fd;
	QString file = fd.getSaveFileName(this, "Save layer stack...",
	                                  fd.directory().absolutePath(),
	                                  PICVIZ_LAYER_ARCHIVE_FILTER ";;" ALL_FILES_FILTER);
	if(!file.isEmpty()) {
		current_view()->get_layer_stack().save_to_file(file);
	}
#endif
}

void PVInspector::PVMainWindow::layer_load_ls_Slot()
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	if (current_view() == nullptr) {
		return;
	}

	QFileDialog fd;
	QString file = fd.getOpenFileName(this, "Import a layer stack...",
	                                  fd.directory().absolutePath(),
	                                  PICVIZ_LAYER_ARCHIVE_FILTER ";;" ALL_FILES_FILTER);
	if(!file.isEmpty()) {
		current_view()->get_layer_stack().load_from_file(file);
	}
#endif
}

void PVInspector::PVMainWindow::layer_copy_ls_details_to_clipboard_Slot()
{
	if (current_view() == nullptr) {
		return;
	}

	current_view()->get_layer_stack().copy_details_to_clipboard();
}

void PVInspector::PVMainWindow::layer_reset_color_Slot()
{
	if (current_view() == nullptr) {
		return;
	}

	Picviz::PVView_sp lib_view(current_view()->shared_from_this());
	if (lib_view) {
		lib_view->get_current_layer().reset_to_default_color();
		PVHive::PVCallHelper::call<FUNC(Picviz::PVView::process_from_layer_stack)>(lib_view);
	}
}

void PVInspector::PVMainWindow::root_modified()
{
	setWindowModified(true);
}
