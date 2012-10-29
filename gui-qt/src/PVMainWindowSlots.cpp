/**
 * \file PVMainWindowSlots.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <pvkernel/core/PVAxesIndexType.h>
#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/core/PVRecentItemsManager.h>
#include <pvkernel/core/PVSerializeArchiveZip.h>
#include <pvkernel/core/PVSerializeArchiveFixError.h>
#include <pvkernel/core/PVVersion.h>

#include <picviz/PVAxisComputation.h>
#include <picviz/widgets/PVAD2GWidget.h>
#include <picviz/PVPlotting.h>
#include <picviz/PVMapping.h>

#include <picviz/widgets/editors/PVAxisIndexEditor.h>

#include <pvhive/PVHive.h>

#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>

#include <pvguiqt/PVAxesCombinationDialog.h>
#include <pvguiqt/PVLayerFilterProcessWidget.h>
#include <pvguiqt/PVImportSourceToProjectDlg.h>
#include <pvguiqt/PVWorkspace.h>

#include <PVMainWindow.h>
#include <PVExpandSelDlg.h>
#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <picviz/widgets/PVArgumentListWidgetFactory.h>
#include <PVFormatBuilderWidget.h>
#include <PVExtractorWidget.h>
#include <PVSaveSceneDialog.h>
#include <PVAxisComputationDlg.h>

#include <pvguiqt/PVAboutBoxDialog.h>

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
	/*
	PVAxisPropertiesWidget* dlg = current_tab->get_axes_properties_widget(current_tab);
	if (dlg->isVisible()) {
		return;
	}
	dlg->show();*/
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

	// FIXME!!! Why is this broadcasted to every PVGL::PVView for every Picviz::PVView? Shouldn't it be reserved to the _current_ Picviz::PVView ?
	/*PVSDK::PVMessage message;

	message.function = PVSDK_MESSENGER_FUNCTION_TOGGLE_DISPLAY_EDGES;
	pvsdk_messenger->post_message_to_gl(message);*/
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::change_of_current_view_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::change_of_current_view_Slot()
{
#if 0
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);
	/* we set current_tab to it's new value */
	current_tab = dynamic_cast<PVTabSplitter*>(_workspaces_tab_widget->currentWidget());
	if(current_tab!=0){
		connect(current_tab,SIGNAL(selection_changed_signal(bool)),this,SLOT(enable_menu_filter_Slot(bool)));
		current_tab->updateFilterMenuEnabling();
	}
	if (!current_view()) {
		// PVLOG_ERROR("PVInspector::PVMainWindow::%s We have a strange beast in the tab widget: %p!\n", __FUNCTION__, pv_WorkspacesTabWidget->currentWidget());
		menu_activate_is_file_opened(false);
	}
	/* we emit a broadcast signal to spread the news ! */
	emit change_of_current_view_Signal(); // FIXME! I think nobody care about this broadcast!
#endif
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
 * PVInspector::PVMainWindow::lines_display_unselected_listing_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::lines_display_unselected_listing_Slot()
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
 * PVInspector::PVMainWindow::lines_display_unselected_GLview_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::lines_display_unselected_GLview_Slot()
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
	//update_pvglview(current_lib_view, PVSDK_MESSENGER_REFRESH_SELECTION);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::lines_display_zombies_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::lines_display_zombies_Slot()
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
	//update_pvglview(current_lib_view, PVSDK_MESSENGER_REFRESH_SELECTION);
	/* We refresh the listing */
	// TODO: hive!
	//current_tab->update_pv_listing_model_Slot();

}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::lines_display_zombies_listing_Sloupdate_recent_projectst()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::lines_display_zombies_listing_Slot()
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
 * PVInspector::PVMainWindow::lines_display_zombies_GLview_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::lines_display_zombies_GLview_Slot()
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
	//update_pvglview(current_lib_view, PVSDK_MESSENGER_REFRESH_SELECTION);
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

	if (axes.size() > 0) {
		//update_pvglview(cur_view_p, PVSDK_MESSENGER_REFRESH_POSITIONS);
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
	
	QFile file;
	while (true) {
		QString filename = pv_ExportSelectionDialog->getSaveFileName();
		if (filename.isEmpty()) {
			return;
		}

		file.setFileName(filename);
		if (!file.open(QIODevice::WriteOnly)) {
			QMessageBox err(QMessageBox::Critical, tr("Error while writing the selection"), tr("Unable to write the selection to %1").arg(filename));
			err.exec();
		}
		else {
			break;
		}
	}

	setCursor(Qt::WaitCursor);

	// TODO: put an option in the widget for the file locale
	// Open a text stream with the current locale (by default in QTextStream)
	QTextStream stream(&file);

	// For now, save the NRAW !
	Picviz::PVView* view = current_view();
	PVRush::PVNraw const& nraw = view->get_rushnraw_parent();
	view->get_real_output_selection().write_selected_lines_nraw(stream, nraw, 0);

	setCursor(Qt::ArrowCursor);

	QMessageBox end(QMessageBox::Information, tr("Export selection"), tr("The selection has been successfully written."));
	end.exec();
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
	//current_tab->get_extractor_widget()->refresh_and_show();
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

PVInspector::PVMainWindow* PVInspector::PVMainWindow::find_main_window(QString const& file)
{
	// From Qt's example...
	QString canonicalFilePath = QFileInfo(file).canonicalFilePath();

	foreach (QWidget *widget, qApp->topLevelWidgets()) {
		PVMainWindow *mainWin = qobject_cast<PVMainWindow *>(widget);
		if (mainWin && mainWin->_cur_project_file == canonicalFilePath) {
			return mainWin;
		}
	}
	return NULL;
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::project_new_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::project_new_Slot()
{
	PVCore::PVDataTreeAutoShared<Picviz::PVScene> scene_p = PVCore::PVDataTreeAutoShared<Picviz::PVScene>(get_root_sp(), _cur_project_file);
	scene_p->set_name(tr("new-project%1." PICVIZ_SCENE_ARCHIVE_EXT).arg(sequence_n++));
	_projects_tab_widget->add_project(scene_p);
}

void PVInspector::PVMainWindow::load_source_from_description_Slot(PVRush::PVSourceDescription src_desc)
{
	if (_projects_tab_widget->projects_count() == 0) {
		// No loaded project: create a new one and load the source
		PVCore::PVDataTreeAutoShared<Picviz::PVScene> scene_p = PVCore::PVDataTreeAutoShared<Picviz::PVScene>(get_root_sp(), _cur_project_file);
		Picviz::PVSource_p src_p = Picviz::PVSource::create_source_from_description(scene_p, src_desc);
		load_source(src_p);
	}
	else if (_projects_tab_widget->projects_count() == 1) {
		// Only one project loaded: use it to load the source
		Picviz::PVSource_p src_p = Picviz::PVSource::create_source_from_description(current_scene()->shared_from_this(), src_desc);
		load_source(src_p);
	}
	else {
		// More than one project loaded: ask the user the project he wants to use to load the source
		PVGuiQt::PVImportSourceToProjectDlg dlg(_projects_tab_widget->get_projects_list(), _projects_tab_widget->get_current_project_index());
		if (dlg.exec() == QDialog::Accepted) {
			int project_index = dlg.result();
			select_scene(project_index);
			Picviz::PVSource_p src_p = Picviz::PVSource::create_source_from_description(current_scene()->shared_from_this(), src_desc);
			load_source(src_p);
		}
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::project_load_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::project_load_Slot()
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	_load_project_dlg.setFileMode(QFileDialog::ExistingFile);
	_load_project_dlg.setAcceptMode(QFileDialog::AcceptOpen);
	if (_load_project_dlg.exec() != QDialog::Accepted) {
		return;
	}
	QString file = _load_project_dlg.selectedFiles().at(0);

	load_project(file);
#endif
}

void PVInspector::PVMainWindow::create_new_window_for_workspace(QWidget* widget_workspace)
{
	PVMainWindow* other = new PVMainWindow();
	other->move(QCursor::pos());
	other->resize(size());
	other->show();

	other->menu_activate_is_file_opened(true);
	other->show_start_page(false);
	//other->_workspaces_tab_widget->setVisible(true);
	other->set_project_modified(true);

	PVGuiQt::PVWorkspace* workspace = dynamic_cast<PVGuiQt::PVWorkspace*>(widget_workspace);
	if (workspace) {
		_projects_tab_widget->remove_workspace(workspace, false);
		other->_projects_tab_widget->add_workspace((PVGuiQt::PVWorkspace*) workspace);
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

bool PVInspector::PVMainWindow::load_project(QString const& file)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	/*if (!maybe_save_project()) {
		return false;
	}
	set_project_modified(false);*/

	/*close_scene();*/


	Picviz::PVScene* scene = _projects_tab_widget->get_scene_from_path(file);

	if (scene) {
		_projects_tab_widget->select_project(scene);
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

	bool project_has_been_fixed = false;
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
				project_has_been_fixed = true;
				close_scene();
				//_scene.reset(new Picviz::PVScene("root", root.get()));
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

	menu_activate_is_file_opened(true);
	show_start_page(false);
	//_workspaces_tab_widget->setVisible(true);

	set_current_project_filename(file);
	if (project_has_been_fixed) {
		set_project_modified(true);
	}

	PVHive::call<FUNC(PVCore::PVRecentItemsManager::add)>(PVCore::PVRecentItemsManager::get(), file, PVCore::PVRecentItemsManager::Category::PROJECTS);
#endif

	return true;
}

void PVInspector::PVMainWindow::set_current_project_filename(QString const& file)
{
	static int sequence_n = 1;

	_is_project_untitled = file.isEmpty();

	if (is_project_untitled()) {
		_cur_project_file = tr("new-project%1." PICVIZ_SCENE_ARCHIVE_EXT).arg(sequence_n);
		sequence_n++;
	}
	else {
		_cur_project_file = QFileInfo(file).canonicalFilePath();
	}

	setWindowTitle(QString());
	set_project_modified(false);
	setWindowFilePath(_cur_project_file);
}

void PVInspector::PVMainWindow::set_project_modified(bool modified)
{
	setWindowModified(modified);
#ifdef CUSTOMER_CAPABILITY_SAVE
	project_save_Action->setEnabled(modified);
#endif
}

void PVInspector::PVMainWindow::project_modified_Slot()
{
	set_project_modified(true);
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
	if (current_view()) {
		PVCore::PVSerializeArchiveOptions_p options(current_scene()->get_default_serialize_options());
		PVSaveSceneDialog* dlg = new PVSaveSceneDialog(current_scene()->shared_from_this(), options, this);
		if (!_current_save_project_folder.isEmpty()) {
			dlg->setDirectory(_current_save_project_folder);
		}
		dlg->selectFile(_cur_project_file);
		if (dlg->exec() == QDialog::Accepted) {
			QString file = dlg->selectedFiles().at(0);
			ret = save_project(file, options);
		}
		_current_save_project_folder = dlg->directory().absolutePath();
		dlg->deleteLater();
	}
#endif
	return ret;
}

bool PVInspector::PVMainWindow::save_project(QString const& file, PVCore::PVSerializeArchiveOptions_p options)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	try {
		current_scene()->save_to_file(file, options);
	}
	catch (PVCore::PVSerializeArchiveError& e) {
		QMessageBox* box = new QMessageBox(QMessageBox::Critical, tr("Error while saving project..."), tr("Error while saving project %1:\n%2").arg(file).arg(e.what()), QMessageBox::Ok, this);
		box->exec();
		return false;
	}

	set_current_project_filename(file);

	PVHive::call<FUNC(PVCore::PVRecentItemsManager::add)>(PVCore::PVRecentItemsManager::get(), file, PVCore::PVRecentItemsManager::Category::PROJECTS);

	return true;
#else
	return false;
#endif
}

bool PVInspector::PVMainWindow::maybe_save_project()
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	if (isWindowModified()) {
		QMessageBox::StandardButton ret;
		ret = QMessageBox::warning(this, tr("Picviz Inspector"),
				tr("The project has been modified.\n"
					"Do you want to save your changes?"),
				QMessageBox::Save | QMessageBox::Discard
				| QMessageBox::Cancel);
		if (ret == QMessageBox::Save) {
			return project_save_Slot();
		}
		if (ret == QMessageBox::Cancel) {
			return false;
		}
	}
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

/*void PVInspector::PVMainWindow::view_new_parallel_Slot()
{
	PVLOG_INFO("PVInspector::PVMainWindow::%s\n", __FUNCTION__);


	QDialog* dlg = new QDialog(this);
	PVHive::PVObserverSignal<Picviz::PVView>* new_obs = new PVHive::PVObserverSignal<Picviz::PVView>(dlg);
	new_obs->connect_about_to_be_deleted(dlg, SLOT(reject()));
	dlg->setAttribute(Qt::WA_DeleteOnClose, true);

	QLayout *layout = new QVBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	dlg->setLayout(layout);
	PVParallelView::PVLibView* parallel_lib_view;

	// Progress box!
	PVCore::PVProgressBox* pbox_lib = new PVCore::PVProgressBox("Creating new view...", (QWidget*) this);
	pbox_lib->set_enable_cancel(false);
	PVCore::PVProgressBox::progress<PVParallelView::PVLibView*>(boost::bind(&PVParallelView::common::get_lib_view, boost::ref(*current_view())), pbox_lib, parallel_lib_view);

	QWidget *view = parallel_lib_view->create_view(dlg);
	layout->addWidget(view);

	Picviz::PVView_sp view_sp(current_view()->shared_from_this());
	PVHive::get().register_observer(view_sp, *new_obs);
	dlg->show();
}*/

void PVInspector::PVMainWindow::view_new_parallel_Slot()
{
	PVLOG_INFO("PVInspector::PVMainWindow::%s\n", __FUNCTION__);
	QDialog* dlg = new QDialog(this);

	QLayout *layout = new QVBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	dlg->setLayout(layout);
	PVParallelView::PVLibView* parallel_lib_view;

	// Progress box!
	PVCore::PVProgressBox* pbox_lib = new PVCore::PVProgressBox("Creating new view...", (QWidget*) this);
	pbox_lib->set_enable_cancel(false);
	PVCore::PVProgressBox::progress<PVParallelView::PVLibView*>(boost::bind(&PVParallelView::common::get_lib_view, boost::ref(*current_view())), pbox_lib, parallel_lib_view);

	QWidget *view = parallel_lib_view->create_view();

	layout->addWidget(view);

	dlg->show();
}

void PVInspector::PVMainWindow::view_new_zoomed_parallel_Slot()
{
	PVLOG_INFO("PVInspector::PVMainWindow::%s\n", __FUNCTION__);

	QDialog *dlg = new QDialog(this);
	dlg->setModal(true);

	QLayout *layout = new QVBoxLayout();
	dlg->setLayout(layout);

	QLabel *label = new QLabel("Open a zoomed view on axis:");
	layout->addWidget(label);

	PVWidgets::PVAxisIndexEditor *axes = new PVWidgets::PVAxisIndexEditor(*current_view(), dlg);
	axes->set_axis_index(0);
	layout->addWidget(axes);

	QDialogButtonBox *dbb = new QDialogButtonBox(QDialogButtonBox::Open | QDialogButtonBox::Cancel);

	QObject::connect(dbb, SIGNAL(accepted()), dlg, SLOT(accept()));
	QObject::connect(dbb, SIGNAL(rejected()), dlg, SLOT(reject()));

	layout->addWidget(dbb);

	if (dlg->exec() == QDialog::Accepted) {
		QDialog *view_dlg = new QDialog();
		PVHive::PVObserverSignal<Picviz::PVView>* new_obs = new PVHive::PVObserverSignal<Picviz::PVView>(view_dlg);
		new_obs->connect_about_to_be_deleted(view_dlg, SLOT(reject()));

		view_dlg->setMaximumWidth(1024);
		view_dlg->setMaximumHeight(1024);
		view_dlg->setAttribute(Qt::WA_DeleteOnClose, true);

		QLayout *view_layout = new QVBoxLayout(view_dlg);
		view_layout->setContentsMargins(0, 0, 0, 0);
		view_dlg->setLayout(view_layout);

		int axis_index = axes->get_axis_index().get_original_index();

		PVParallelView::PVLibView* parallel_lib_view;

		// Progress box!
		PVCore::PVProgressBox* pbox_lib = new PVCore::PVProgressBox("Creating new view...", (QWidget*) this);
		pbox_lib->set_enable_cancel(false);
		PVCore::PVProgressBox::progress<PVParallelView::PVLibView*>(boost::bind(&PVParallelView::common::get_lib_view, boost::ref(*current_view())), pbox_lib, parallel_lib_view);

		QWidget *view = parallel_lib_view->create_zoomed_view(axis_index);

		Picviz::PVView_sp view_sp(current_view()->shared_from_this());
		PVHive::get().register_observer(view_sp, *new_obs);

		view_layout->addWidget(view);
		view_dlg->show();
	}

	dlg->deleteLater();

}

void PVInspector::PVMainWindow::view_screenshot_qt_Slot()
{

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
	connect(editorWidget, SIGNAL(accepted()), this, SLOT(cur_format_changed_Slot()));
	connect(editorWidget, SIGNAL(rejected()), this, SLOT(cur_format_changed_Slot()));
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

void PVInspector::PVMainWindow::cur_format_changed_Slot()
{
	PVFormatBuilderWidget* editor = dynamic_cast<PVFormatBuilderWidget*>(sender());
	assert(editor);
	PVTabSplitter* src_tab = dynamic_cast<PVTabSplitter*>(editor->parent());
	assert(src_tab);
	Picviz::PVSource const* cur_src = src_tab->get_lib_src();

	PVRush::PVFormat old_format = cur_src->get_format();
	PVRush::PVFormat new_format(old_format.get_format_name(), old_format.get_full_path());
	new_format.populate();

	PVRush::PVFormat::Comparaison comp = new_format.comp(old_format);
	if (comp.same()) {
		return;
	}

#if 0
	// Too unstable, because it does not take into account the fact that the axes could have completely changed.
	// We should recreate a new PVSource !
	if (comp.need_extract()) {
		QMessageBox* box = new QMessageBox(QMessageBox::Question, tr("Format modified"), tr("The splitters and/or filters of this format have been changed. Do you want to reextract your data ?"), QMessageBox::Yes | QMessageBox::No, this);
		if (box->exec() == QMessageBox::Yes) {
			PVRush::PVExtractor& extractor = cur_src->get_extractor();
			extractor.save_nraw();
			PVRush::PVControllerJob_p job = cur_src->extract();
			src_tab->process_extraction_job(job);
			return;
		}
	}

	if (!comp.need_extract() && (comp.different_mapping() || comp.different_plotting())) {
		QMessageBox* box = new QMessageBox(QMessageBox::Question, tr("Format modified"), tr("The mapping and/or plotting properties of this format have been changed. Do you want to update the current view ?"), QMessageBox::Yes | QMessageBox::No, this);
		if (box->exec() == QMessageBox::Yes) {
			Picviz::PVView* cur_view = cur_src->current_view();
			Picviz::PVMapped* mapped = cur_view->get_parent<Picviz::PVMapped>();
			Picviz::PVPlotted* plotted = cur_view->get_parent<Picviz::PVPlotted>();
			mapped->get_mapping()->reset_from_format(new_format);
			plotted->get_plotting().reset_from_format(new_format);
			if (comp.different_mapping()) {
				src_tab->process_mapped_if_current(mapped);
			}
			else {
				src_tab->process_plotted_if_current(plotted);
			}
			cur_src->set_format(new_format);
		}
	}
#endif
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
		Picviz::PVView_sp view(current_view());
		set_selection_from_layer(view, view->get_current_layer());
	}
}

void PVInspector::PVMainWindow::selection_set_from_layer_Slot()
{
	if (current_view()) {
		Picviz::PVView_sp view(current_view());

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
	if (!_ad2g_mw) {
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
	_ad2g_mw->exec();
}
