//! \file PVMainWindowSlots.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvkernel/core/PVAxesIndexType.h>
#include <pvkernel/core/PVSerializeArchiveZip.h>
#include <pvkernel/core/PVSerializeArchiveFixError.h>
#include <pvkernel/core/PVVersion.h>
#include <picviz/PVAxisComputation.h>

#include <PVMainWindow.h>
#include <PVExpandSelDlg.h>
#include <PVArgumentListWidget.h>
#include <PVXmlEditorWidget.h>
#include <PVLayerFilterProcessWidget.h>
#include <PVAxesCombinationDialog.h>
#include <PVExtractorWidget.h>
#include <PVSaveSceneDialog.h>
#include <PVAxisComputationDlg.h>

/******************************************************************************
 *
 * PVInspector::PVMainWindow::about_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::about_Slot()
{
	if (!about_dialog) {
		about_dialog = new QDialog;

		QGridLayout *main_layout = new QGridLayout;

		QLabel *logo = new QLabel;
#ifdef CUDA
		QString content = "Picviz Inspector v." + QString(PICVIZ_CURRENT_VERSION_STR) + "\n(c) 2010-2011 Picviz Labs SAS\ncontact@picviz.com\nhttp://www.picviz.com\n\nWith CUDA support\nQT version " + QString(QT_VERSION_STR);
#else
		QString content = "Picviz Inspector v." + QString(PICVIZ_CURRENT_VERSION_STR) + "\n(c) 2010-2011 Picviz Labs SAS\ncontact@picviz.com\nhttp://www.picviz.com\n\nQT version " + QString(QT_VERSION_STR);
#endif
		QLabel *text = new QLabel(content);
		QPushButton *ok = new QPushButton("OK");

		logo->setPixmap(QPixmap(":/logo.png"));

		main_layout->addWidget(logo, 0, 0);
		main_layout->addWidget(text, 0, 1);
		main_layout->addWidget(ok, 2, 1);

		about_dialog->setLayout(main_layout);

		about_dialog->setWindowTitle("About Picviz Inspector");

		about_dialog->connect(ok, SIGNAL(pressed()), about_dialog, SLOT(hide()));
	}
	about_dialog->show();
}

void PVInspector::PVMainWindow::axes_editor_Slot()
{
	if (!current_tab) {
		return;
	}
	PVAxisPropertiesWidget* dlg = current_tab->get_axes_properties_widget(current_tab->get_lib_view());
	if (dlg->isVisible()) {
		return;
	}
	dlg->show();
}

void PVInspector::PVMainWindow::axes_combination_editor_Slot()
{
	if (!current_tab) {
		return;
	}

	PVAxesCombinationDialog* dlg = current_tab->get_axes_combination_editor(current_tab->get_lib_view());
	if (dlg->isVisible()) {
		return;
	}

	dlg->save_current_combination();
	dlg->update_used_axes();
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
	Picviz::PVView_p current_lib_view;

	if (!current_tab) {
		return;
	}
	current_lib_view = current_tab->get_lib_view();

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
	PVSDK::PVMessage message;

	message.function = PVSDK_MESSENGER_FUNCTION_TOGGLE_DISPLAY_EDGES;
	pvsdk_messenger->post_message_to_gl(message);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::change_of_current_view_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::change_of_current_view_Slot()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);
	/* we set current_tab to it's new value */
	current_tab = dynamic_cast<PVTabSplitter*>(pv_ListingsTabWidget->currentWidget());
	if(current_tab!=0){
		connect(current_tab,SIGNAL(selection_changed_signal(bool)),this,SLOT(enable_menu_filter_Slot(bool)));
		current_tab->updateFilterMenuEnabling();
	}
	if (!current_tab) {
		// PVLOG_ERROR("PVInspector::PVMainWindow::%s We have a strange beast in the tab widget: %p!\n", __FUNCTION__, pv_ListingsTabWidget->currentWidget());
		menu_activate_is_file_opened(false);
	}
	/* we emit a broadcast signal to spread the news ! */
	emit change_of_current_view_Signal(); // FIXME! I think nobody care about this broadcast!
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::commit_selection_in_current_layer_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::commit_selection_in_current_layer_Slot()
{
	/* We prepare a direct access to the current lib_view */
	Picviz::PVView_p current_lib_view;

	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);

	if (pv_ListingsTabWidget->currentIndex() == -1) {
		return;
	}
	current_lib_view = current_tab->get_lib_view();
	commit_selection_in_current_layer(current_lib_view);
}

/******************************************************************************
 *
* PVInspector::PVMainWindow::commit_selection_to_new_layer_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::commit_selection_to_new_layer_Slot()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);
	/* VARIABLES */
	/* We prepare a direct access to the current lib_view */
	Picviz::PVView_p current_lib_view;

	/* CODE */
	if (pv_ListingsTabWidget->currentIndex() == -1) {
		return;
	}
	current_lib_view = current_tab->get_lib_view();
	commit_selection_to_new_layer(current_lib_view);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::lines_display_unselected_listing_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::lines_display_unselected_listing_Slot()
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

	state_machine->toggle_listing_unselected_visibility();
	/* We refresh the listing */
	current_tab->update_pv_listing_model_Slot();

	if (!lines_display_unselected_listing_Action->text().compare(QString(tr("Hide unselected lines in listing")))) {
		lines_display_unselected_listing_Action->setText(QString(tr("Display unselected lines in listing")));
	} else {
		lines_display_unselected_listing_Action->setText(QString(tr("Hide unselected lines in listing")));
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::lines_display_unselected_GLview_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::lines_display_unselected_GLview_Slot()
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
	/* We refresh the view */
	current_lib_view->process_visibility();
	update_pvglview(current_lib_view, PVSDK_MESSENGER_REFRESH_SELECTION);

	if (!lines_display_unselected_GLview_Action->text().compare(QString(tr("Hide unselected lines in view")))) {
		lines_display_unselected_GLview_Action->setText(QString(tr("Display unselected lines in view")));
	} else {
		lines_display_unselected_GLview_Action->setText(QString(tr("Hide unselected lines in view")));
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::lines_display_zombies_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::lines_display_zombies_Slot()
{
	Picviz::PVView_p current_lib_view;
	Picviz::PVStateMachine *state_machine = NULL;

	if (!current_tab) {
		return;
	}
	current_lib_view = current_tab->get_lib_view();
	state_machine = current_lib_view->state_machine;

	state_machine->toggle_listing_zombie_visibility();
	state_machine->toggle_gl_zombie_visibility();
	/* We set the listing to be the same */
	// state_machine->set_listing_zombie_visibility(state_machine->are_zombie_visible());
	/* We refresh the view */
	current_lib_view->process_visibility();
	update_pvglview(current_lib_view, PVSDK_MESSENGER_REFRESH_SELECTION);
	/* We refresh the listing */
	current_tab->update_pv_listing_model_Slot();

	if (!lines_display_zombies_Action->text().compare(QString(tr("Hide zombies lines")))) {
		lines_display_zombies_Action->setText(QString(tr("Display zombies lines")));
	} else {
		lines_display_zombies_Action->setText(QString(tr("Hide zombies lines")));
	}

}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::lines_display_zombies_listing_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::lines_display_zombies_listing_Slot()
{
	Picviz::PVView_p current_lib_view;
	Picviz::PVStateMachine *state_machine = NULL;

	if (!current_tab) {
		return;
	}
	current_lib_view = current_tab->get_lib_view();
	state_machine = current_lib_view->state_machine;

	state_machine->toggle_listing_zombie_visibility();
	/* We refresh the listing */
	current_tab->update_pv_listing_model_Slot();

	if (!lines_display_zombies_listing_Action->text().compare(QString(tr("Hide zombies lines in listing")))) {
		lines_display_zombies_listing_Action->setText(QString(tr("Display zombies lines in listing")));
	} else {
		lines_display_zombies_listing_Action->setText(QString(tr("Hide zombies lines in listing")));
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::lines_display_zombies_GLview_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::lines_display_zombies_GLview_Slot()
{
	Picviz::PVView_p current_lib_view;
	Picviz::PVStateMachine *state_machine = NULL;

	if (!current_tab) {
		return;
	}
	current_lib_view = current_tab->get_lib_view();
	state_machine = current_lib_view->state_machine;

	state_machine->toggle_gl_zombie_visibility();
	/* We refresh the view */
	current_lib_view->process_visibility();
	update_pvglview(current_lib_view, PVSDK_MESSENGER_REFRESH_SELECTION);

	if (!lines_display_zombies_GLview_Action->text().compare(QString(tr("Hide zombies lines in view")))) {
		lines_display_zombies_GLview_Action->setText(QString(tr("Display zombies lines in view")));
	} else {
		lines_display_zombies_GLview_Action->setText(QString(tr("Hide zombies lines in view")));
	}
}

void PVInspector::PVMainWindow::expand_selection_on_axis_Slot()
{
	if (!current_tab) {
		return;
	}
	Picviz::PVView_p cur_view_p = current_tab->get_lib_view();
	PVExpandSelDlg* dlg = new PVExpandSelDlg(cur_view_p, this);
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
		update_pvglview(cur_view_p, PVSDK_MESSENGER_REFRESH_POSITIONS);
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
	Picviz::PVView_p view = current_tab->get_lib_view();
	PVRush::PVNraw const& nraw = view->get_rushnraw_parent();
	view->get_real_output_selection().write_selected_lines_nraw(stream, nraw);

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

	if (!current_tab || !current_tab->get_lib_view()) {
		return;
	}
	/* We do all that has to be done in the lib FIRST */
	current_tab->get_lib_view()->apply_filter_named_select_all();
	current_tab->get_lib_view()->process_from_eventline();

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
	if (current_tab && current_tab->get_lib_view()) {
		QObject *s = sender();
		Picviz::PVView_p lib_view = current_tab->get_lib_view();
		QString filter_name = s->objectName();

		//get filter
		Picviz::PVLayerFilter::p_type filter_org = LIB_CLASS(Picviz::PVLayerFilter)::get().get_class_by_name(filter_name);
		//cpy filter
		Picviz::PVLayerFilter::p_type fclone = filter_org->clone<Picviz::PVLayerFilter>();
		PVCore::PVArgumentList &args = lib_view->filters_args[filter_name];
		PVLayerFilterProcessWidget* filter_widget = new PVLayerFilterProcessWidget(current_tab, args, fclone);
		filter_widget->show();
	}
}

void PVInspector::PVMainWindow::extractor_file_Slot()
{
	if (!current_tab) {
		//TODO: this should not happen because the menu item should be disabled... !
		return;
	}
	current_tab->get_extractor_widget()->refresh_and_show();
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
	if (maybe_save_project()) {
		close_scene();
		set_current_project_filename(QString());
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
	QFileDialog* dlg = new QFileDialog(this, tr("Load a project..."), QString(), PICVIZ_SCENE_ARCHIVE_FILTER ";;" ALL_FILES_FILTER);
	dlg->setFileMode(QFileDialog::ExistingFile);
	dlg->setAcceptMode(QFileDialog::AcceptOpen);
	if (dlg->exec() != QDialog::Accepted) {
		return;
	}
	QString file = dlg->selectedFiles().at(0);

	load_project(file);
#if 0
	PVMainWindow* existing = find_main_window(file);
	if (existing) {
		existing->show();
		existing->raise();
		existing->activateWindow();
		return;
	}
	if (is_project_untitled() && _scene->is_empty() && !isWindowModified()) {
		load_project(file);
	}
	else {
		PVMainWindow* other = new PVMainWindow();
		if (!other->load_project(file)) {
			other->deleteLater();
			return;
		}
		other->move(x() + 40, y() + 40);
		other->show();
	}
#endif

#endif
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
	if (!maybe_save_project()) {
		return false;
	}
	set_project_modified(false);

	close_scene();
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
			_scene->load_from_archive(ar);
		}
		catch (PVCore::PVSerializeArchiveError& e) {
			err_msg = tr("Error while loading project %1:\n%2").arg(file).arg(e.what());
		}
		catch (PVRush::PVInputException const& e)
		{
			err_msg = tr("Error while loading project %1:\n%2").arg(file).arg(QString::fromStdString(e.what()));
		}
		if (ar->has_repairable_errors()) {
			if (fix_project_errors(ar)) {
				project_has_been_fixed = true;
				_scene.reset(new Picviz::PVScene("root", root.get()));
				continue;
			}
			else {
				if (!err_msg.isEmpty()) {
					QMessageBox* box = new QMessageBox(QMessageBox::Critical, tr("Error while loading project..."), err_msg, QMessageBox::Ok, this);
					box->exec();
				}
				_scene.reset();
				return false;
			}
		}
		break;
	}

	if (!load_scene()) {
		PVLOG_ERROR("(PVMainWindow::project_load_Slot) error while processing the scene...\n");
		_scene.reset();
		return false;
	}

	menu_activate_is_file_opened(true);
	show_start_page(false);
	pv_ListingsTabWidget->setVisible(true);

	set_current_project_filename(file);
	if (project_has_been_fixed) {
		set_project_modified(true);
	}
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

	set_project_modified(false);
	setWindowTitle(QString());
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
		PVCore::PVSerializeArchiveOptions_p options(_scene->get_default_serialize_options());
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
	if (current_tab && current_tab->get_lib_view()) {
		PVCore::PVSerializeArchiveOptions_p options(_scene->get_default_serialize_options());
		PVSaveSceneDialog* dlg = new PVSaveSceneDialog(_scene, options, this);
		dlg->selectFile(_cur_project_file);
		if (dlg->exec() == QDialog::Accepted) {
			QString file = dlg->selectedFiles().at(0);
			ret = save_project(file, options);
		}	
		dlg->deleteLater();
	}
#endif
	return ret;
}

bool PVInspector::PVMainWindow::save_project(QString const& file, PVCore::PVSerializeArchiveOptions_p options)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	try {
		_scene->save_to_file(file, options);
	}
	catch (PVCore::PVSerializeArchiveError& e) {
		QMessageBox* box = new QMessageBox(QMessageBox::Critical, tr("Error while saving project..."), tr("Error while saving project %1:\n%2").arg(file).arg(e.what()), QMessageBox::Ok, this);
		box->exec();
		return false;
	}

	set_current_project_filename(file);

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
	if (!current_tab) {
		return;
	}

	Picviz::PVView_p lib_view = current_tab->get_lib_view();
	if (lib_view) {
		lib_view->select_all_nonzb_lines();
		// Set square area mode w/ volatile
		update_pvglview(lib_view, PVSDK_MESSENGER_REFRESH_SELECTION);
		current_tab->refresh_listing_Slot();
		current_tab->updateFilterMenuEnabling();
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
	if (!current_tab) {
		return;
	}

	Picviz::PVView_p lib_view = current_tab->get_lib_view();
	if (lib_view) {
		lib_view->select_no_line();
		update_pvglview(lib_view, PVSDK_MESSENGER_REFRESH_SELECTION);
		current_tab->refresh_listing_Slot();
		current_tab->updateFilterMenuEnabling();
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
	if (!current_tab) {
		return;
	}

	Picviz::PVView_p lib_view = current_tab->get_lib_view();
	if (lib_view) {
		lib_view->select_inv_lines();
		update_pvglview(lib_view, PVSDK_MESSENGER_REFRESH_SELECTION);
		current_tab->refresh_listing_Slot();
		current_tab->updateFilterMenuEnabling();
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
	if (!current_tab || !current_tab->get_lib_view())
		return;
	set_color(current_tab->get_lib_view());
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

		current_tab->get_lib_view()->apply_filter_from_name(last_sendername, args);
		current_tab->get_lib_view()->process_from_eventline();

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

	// Ask the PVGL to create a GL-View of the currently selected view.
	if (current_tab && current_tab->get_lib_view()) {
		PVSDK::PVMessage message;

		message.function = PVSDK_MESSENGER_FUNCTION_CREATE_SCATTER_VIEW;
		message.pv_view = current_tab->get_lib_view();
		message.pointer_1 = new QString(current_tab->get_current_view_name());
		pvsdk_messenger->post_message_to_gl(message);
	}
}

void PVInspector::PVMainWindow::view_new_parallel_Slot()
{
	PVLOG_INFO("PVInspector::PVMainWindow::%s\n", __FUNCTION__);

	// Ask the PVGL to create a GL-View of the currently selected view. 
	if (current_tab && current_tab->get_lib_view()) {
		PVSDK::PVMessage message;

		message.function = PVSDK_MESSENGER_FUNCTION_PLEASE_WAIT;
		message.pointer_1 = new QString(current_tab->get_current_view_name());
		pvsdk_messenger->post_message_to_gl(message);

		message.function = PVSDK_MESSENGER_FUNCTION_CREATE_VIEW;
		message.pv_view = current_tab->get_lib_view();
		message.pointer_1 = new QString(current_tab->get_current_view_name());
		pvsdk_messenger->post_message_to_gl(message);
	}
}

void PVInspector::PVMainWindow::view_screenshot_qt_Slot()
{
	// Get a QImage of the current view
	PVSDK::PVMessage message;
	message.pv_view = current_tab->get_lib_view(); // Get current view
	message.function = PVSDK_MESSENGER_FUNCTION_TAKE_SCREENSHOT;
	message.int_1 = -1;
	message.int_2 = false;
	QImage* image = new QImage();
	message.pointer_1 = image;
	pvsdk_messenger->post_message_to_gl(message);
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
    PVXmlEditorWidget *editorWidget = new PVXmlEditorWidget(this);
    editorWidget->show();
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::cur_format_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::cur_format_Slot()
{
	Picviz::PVSource_p cur_src = current_tab->get_lib_src();
	if (!current_tab || !cur_src) {
		return;
	}
	PVRush::PVFormat const& format = cur_src->get_format();
	if (format.get_full_path().isEmpty()) {
		return;
	}

    PVXmlEditorWidget *editorWidget = new PVXmlEditorWidget(current_tab);
	connect(editorWidget, SIGNAL(accepted()), this, SLOT(cur_format_changed_Slot()));
	connect(editorWidget, SIGNAL(rejected()), this, SLOT(cur_format_changed_Slot()));
	editorWidget->openFormat(format.get_full_path());
    editorWidget->show();
}

void PVInspector::PVMainWindow::cur_format_changed_Slot()
{
	PVXmlEditorWidget* editor = dynamic_cast<PVXmlEditorWidget*>(sender());
	assert(editor);
	PVTabSplitter* src_tab = dynamic_cast<PVTabSplitter*>(editor->parent());
	assert(src_tab);
	Picviz::PVSource_p cur_src = src_tab->get_lib_src();

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
#endif

	if (!comp.need_extract() && (comp.different_mapping() || comp.different_plotting())) {
		QMessageBox* box = new QMessageBox(QMessageBox::Question, tr("Format modified"), tr("The mapping and/or plotting properties of this format have been changed. Do you want to update the current view ?"), QMessageBox::Yes | QMessageBox::No, this);
		if (box->exec() == QMessageBox::Yes) {
			Picviz::PVView_p cur_view = cur_src->current_view();
			Picviz::PVMapped* mapped = cur_view->get_mapped_parent();
			Picviz::PVPlotted* plotted = cur_view->get_plotted_parent();
			mapped->get_mapping().reset_from_format(new_format);
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
}

PVSDK::PVMessenger* PVInspector::PVMainWindow::get_pvmessenger()
{
	return pvsdk_messenger;
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
    PVXmlEditorWidget *editorWidget = new PVXmlEditorWidget(parent);
    editorWidget->show();
	editorWidget->openFormat(path);
}

void PVInspector::PVMainWindow::edit_format_Slot(QDomDocument& doc, QWidget* parent)
{
    PVXmlEditorWidget *editorWidget = new PVXmlEditorWidget(parent);
    editorWidget->show();
	editorWidget->openFormat(doc);
}

void PVInspector::PVMainWindow::axes_new_Slot()
{
	if (!current_tab) {
		return;
	}
	
	Picviz::PVView_p view = current_tab->get_lib_view();
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

	view->get_source_parent()->add_column(ac_plugin->f(), axis);
}
