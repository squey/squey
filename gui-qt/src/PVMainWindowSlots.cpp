/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/qobject_helpers.h>

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/core/PVRecentItemsManager.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/core/PVSerializeArchiveZip.h>
#include <pvkernel/core/PVSerializeArchiveFixError.h>
#include <pvkernel/core/PVVersion.h>
#include <pvkernel/core/PVConfig.h>

#include <inendi/PVPlotting.h>
#include <inendi/PVMapping.h>

#ifdef WITH_MINESET
#include <inendi/PVMineset.h>
#endif

#include <inendi/widgets/editors/PVAxisIndexEditor.h>

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
#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <inendi/widgets/PVArgumentListWidgetFactory.h>
#include <PVFormatBuilderWidget.h>
#include <PVSaveDataTreeDialog.h>

#include <QPainter>
#include <QDockWidget>
#include <QDesktopServices>
#include <QDesktopWidget>

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

void PVInspector::PVMainWindow::axes_combination_editor_Slot()
{
	if (!current_view()) {
		return;
	}

	PVGuiQt::PVAxesCombinationDialog* dlg =
	    ((PVGuiQt::PVWorkspaceBase*)_projects_tab_widget->current_workspace())
	        ->get_axes_combination_editor(current_view());
	if (dlg->isVisible()) {
		return;
	}

	dlg->reset_used_axes();
	dlg->show();
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

	Inendi::PVView* current_lib_view = current_view();
	commit_selection_to_new_layer(current_lib_view);
}

void PVInspector::PVMainWindow::move_selection_to_new_layer_Slot()
{
	if (!current_view()) {
		return;
	}

	Inendi::PVView* current_lib_view = current_view();
	move_selection_to_new_layer(current_lib_view);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::events_display_unselected_listing_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::events_display_unselected_listing_Slot()
{
	Inendi::PVView* current_lib_view;

	if (!current_view()) {
		return;
	}
	current_lib_view = current_view();

	/* We refresh the listing */
	Inendi::PVView_sp view_sp = current_lib_view->shared_from_this();
	PVHive::call<FUNC(Inendi::PVView::toggle_listing_unselected_visibility)>(view_sp);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::events_display_zombies_listing_Sloupdate_recent_projectst()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::events_display_zombies_listing_Slot()
{
	Inendi::PVView* current_lib_view;

	if (!current_view()) {
		return;
	}
	current_lib_view = current_view();

	Inendi::PVView_sp view_sp = current_lib_view->shared_from_this();
	PVHive::call<FUNC(Inendi::PVView::toggle_listing_zombie_visibility)>(view_sp);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::events_display_unselected_zombies_parallelview_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::events_display_unselected_zombies_parallelview_Slot()
{
	Inendi::PVView* current_lib_view;

	if (!current_view()) {
		return;
	}
	current_lib_view = current_view();

	/* We refresh the listing */
	Inendi::PVView_sp view_sp = current_lib_view->shared_from_this();
	PVHive::call<FUNC(Inendi::PVView::toggle_view_unselected_zombie_visibility)>(view_sp);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::export_selection_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::export_selection_Slot()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);

	Inendi::PVView* view = current_view();
	Inendi::PVSelection const& sel = view->get_real_output_selection();

	PVGuiQt::PVExportSelectionDlg::export_selection(*view, sel);
}

#ifdef WITH_MINESET
/******************************************************************************
 *
 * PVInspector::PVMainWindow::export_selection_to_mineset_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::export_selection_to_mineset_Slot()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);

	PVCore::PVProgressBox pbox("Exporting data to Mineset...");
	pbox.set_enable_cancel(false);

	PVCore::PVProgressBox::progress(
	    [&]() {
		    try {
			    std::string dataset_url = Inendi::PVMineset::import_dataset(*current_view());
			    current_view()->add_mineset_dataset(dataset_url);
			    QDesktopServices::openUrl(QUrl(dataset_url.c_str()));
		    } catch (const Inendi::PVMineset::mineset_error& e) {
			    emit mineset_error(QString(e.what()));
		    }
		},
	    &pbox);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::mineset_error_slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::mineset_error_slot(QString error_msg)
{
	QMessageBox::critical(this, "Error when exporting current selection to Mineset", error_msg,
	                      QMessageBox::Ok);
}
#endif

/******************************************************************************
 *
 * PVInspector::PVMainWindow::filter_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::filter_Slot(void)
{
	if (current_view()) {
		QObject* s = sender();
		Inendi::PVView* lib_view = current_view();
		QString filter_name = s->objectName();

		Inendi::PVLayerFilter::p_type filter_org =
		    LIB_CLASS(Inendi::PVLayerFilter)::get().get_class_by_name(filter_name);
		Inendi::PVLayerFilter::p_type fclone = filter_org->clone<Inendi::PVLayerFilter>();
		PVCore::PVArgumentList& args = lib_view->get_last_args_filter(filter_name);
		PVGuiQt::PVLayerFilterProcessWidget* filter_widget =
		    new PVGuiQt::PVLayerFilterProcessWidget(current_view(), args, fclone, this);
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
		Inendi::PVView* lib_view = current_view();
		if (!lib_view->is_last_filter_used_valid()) {
			return;
		}
		QString const& filter_name = lib_view->get_last_used_filter();
		Inendi::PVLayerFilter::p_type filter_org =
		    LIB_CLASS(Inendi::PVLayerFilter)::get().get_class_by_name(filter_name);
		Inendi::PVLayerFilter::p_type fclone = filter_org->clone<Inendi::PVLayerFilter>();
		PVCore::PVArgumentList& args = lib_view->get_last_args_filter(filter_name);
		PVGuiQt::PVLayerFilterProcessWidget* filter_widget =
		    new PVGuiQt::PVLayerFilterProcessWidget(current_view(), args, fclone, this);
		filter_widget->show();
		filter_widget->preview_Slot();
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::project_new_Slot
 *
 *****************************************************************************/
Inendi::PVScene& PVInspector::PVMainWindow::project_new_Slot()
{
	std::string scene_name = tr("Data collection %1").arg(sequence_n++).toStdString();
	Inendi::PVScene& scene_p = get_root_sp()->emplace_add_child(scene_name);
	_projects_tab_widget->add_project(scene_p);

	return scene_p;
}

bool PVInspector::PVMainWindow::load_source_from_description_Slot(
    PVRush::PVSourceDescription src_desc)
{
	bool has_error = false;
	Inendi::PVScene* scene_p;

	PVRush::PVFormat format = src_desc.get_format();
	if ((format.exists() == false) || (QFileInfo(format.get_full_path()).isReadable() == false)) {
		QMessageBox::warning(
		    this, tr("Format \"%1\" can not be read").arg(format.get_format_name()),
		    tr("Check that the file \"%1\" exists and is readable").arg(format.get_full_path()));
		return false;
	}

	auto scenes = get_root().get_children();

	bool new_scene = false;
	if (scenes.size() == 0) {
		// No loaded project: create a new one and load the source
		scene_p = &project_new_Slot();
		new_scene = true;
	} else if (scenes.size() == 1) {
		// Only one project loaded: use it to load the source
		scene_p = scenes.front();
		Inendi::PVRoot_sp root_sp = get_root().shared_from_this();
		PVHive::call<FUNC(Inendi::PVRoot::select_scene)>(root_sp, *scene_p);
	} else {
		// More than one project loaded: ask the user the project he wants to use to
		// load the source
		PVGuiQt::PVImportSourceToProjectDlg* dlg =
		    new PVGuiQt::PVImportSourceToProjectDlg(get_root(), get_root().current_scene(), this);
		if (dlg->exec() != QDialog::Accepted) {
			return false;
		}

		Inendi::PVRoot_sp root_sp = get_root().shared_from_this();
		PVHive::call<FUNC(Inendi::PVRoot::select_scene)>(
		    root_sp, *((Inendi::PVScene*)dlg->get_selected_scene()));
		scene_p = current_scene();
		dlg->deleteLater();
	}

	Inendi::PVSource* src_p;
	try {
		src_p = &scene_p->emplace_add_child(src_desc);
	} catch (PVRush::PVFormatException const& e) {
		PVLOG_ERROR("Error with format: %s\n", qPrintable(e.what()));
		has_error = true;
	} catch (PVRush::PVInputException const& e) {
		QMessageBox::critical(this, tr("Fatal error while loading source..."),
		                      tr("Fatal error while loading source: %1").arg(e.what().c_str()));
		has_error = true;
	}

	if (has_error && new_scene) {
		_projects_tab_widget->remove_project(
		    _projects_tab_widget->get_workspace_tab_widget_from_scene(scene_p));
		return false;
	}

	try {
		if (!load_source(src_p)) {
			remove_source(src_p);
			return false;
		}
	} catch (const PVRush::PVFormatNoTimeMapping& e) {
		QMessageBox::critical(
		    this, tr("Fatal error while loading source..."),
		    (std::string("\nNo mapping format specified for axis '") + e.what() + "'").c_str());
		return false;
	}

	return true;
}

void PVInspector::PVMainWindow::solution_new_Slot()
{
	// FIXME : This Windows is a memory leak
	PVMainWindow* new_mw = new PVMainWindow();
	new_mw->move(x() + 40, y() + 40);
	new_mw->show();
	new_mw->set_window_title_with_filename();
}

void PVInspector::PVMainWindow::solution_load_Slot()
{
	_load_solution_dlg.setFileMode(QFileDialog::ExistingFile);
	_load_solution_dlg.setAcceptMode(QFileDialog::AcceptOpen);
	if (_load_solution_dlg.exec() != QDialog::Accepted) {
		return;
	}
	QString file = _load_solution_dlg.selectedFiles().at(0);
	load_solution_and_create_mw(file);
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
	} else {
		// FIXME : This Windows is a memory leak
		PVMainWindow* other = new PVMainWindow();
		other->move(x() + 40, y() + 40);
		other->show();
		if (!other->load_solution(file)) {
			other->deleteLater();
			return;
		}
	}
}

void PVInspector::PVMainWindow::solution_save_Slot()
{
	if (is_solution_untitled()) {
		solution_saveas_Slot();
	} else {
		PVCore::PVSerializeArchiveOptions_p options(get_root().get_default_serialize_options());
		save_solution(get_solution_path(), options);
	}
}

void PVInspector::PVMainWindow::solution_saveas_Slot()
{
	if (get_root().is_empty()) {
		return;
	}

	PVCore::PVSerializeArchiveOptions_p options(get_root().get_default_serialize_options());
	PVSaveDataTreeDialog* dlg = new PVSaveDataTreeDialog(options, INENDI_ROOT_ARCHIVE_EXT,
	                                                     INENDI_ROOT_ARCHIVE_FILTER, this);
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
}

bool PVInspector::PVMainWindow::maybe_save_solution()
{
	if (isWindowModified()) {
		QMessageBox::StandardButton ret;
		QString solution_name = QFileInfo(windowFilePath()).fileName();
		ret = QMessageBox::warning(this, tr("%1").arg(solution_name),
		                           tr("The solution \"%1\"has been modified.\n"
		                              "Do you want to save your changes?")
		                               .arg(solution_name),
		                           QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
		if (ret == QMessageBox::Save) {
			solution_save_Slot();
			return true;
		}
		if (ret == QMessageBox::Discard) {
			return true;
		} else if (ret == QMessageBox::Cancel) {
			return false;
		}
	}
	return true;
}

bool PVInspector::PVMainWindow::load_solution(QString const& file)
{
	setWindowModified(false);

	PVCore::PVSerializeArchive_p ar;
	PVCore::PVSerializeArchiveError read_exception = PVCore::PVSerializeArchiveError(QString());
	PVCore::PVProgressBox* pbox_solution =
	    new PVCore::PVProgressBox("Loading investigation...", this);
	pbox_solution->set_enable_cancel(true);
	bool ret = PVCore::PVProgressBox::progress(
	    [&] {
		    try {
			    ar.reset(new PVCore::PVSerializeArchiveZip(file, PVCore::PVSerializeArchive::read,
			                                               INENDI_ARCHIVES_VERSION));
		    } catch (const PVCore::PVSerializeArchiveError& e) {
			    read_exception = e;
		    }
		},
	    pbox_solution);
	if (!ret) {
		return false;
	}

	if (!read_exception.what().isEmpty()) {
		QMessageBox* box = new QMessageBox(
		    QMessageBox::Critical, tr("Fatal error while loading solution..."),
		    tr("Fatal error while loading solution %1:\n%2").arg(file).arg(read_exception.what()),
		    QMessageBox::Ok, this);
		box->exec();
		return false;
	}

	bool solution_has_been_fixed = false;
	while (true) {
		QString err_msg;
		try {
			get_root().load_from_archive(ar);
		} catch (PVCore::PVSerializeArchiveError& e) {
			err_msg = tr("Error while loading solution %1:\n%2").arg(file).arg(e.what());
		} catch (PVRush::PVInputException const& e) {
			err_msg = tr("Error while loading solution %1:\n%2")
			              .arg(file)
			              .arg(QString::fromStdString(e.what()));
		} catch (...) {
			err_msg = tr("Fatal error while loading solution %1:\n unhandled error(s).").arg(file);
		}
		if (!err_msg.isEmpty()) {
			QMessageBox* box =
			    new QMessageBox(QMessageBox::Critical, tr("Fatal error while loading solution..."),
			                    err_msg, QMessageBox::Ok, this);
			box->exec();
			return false;
		}
		if (ar->has_repairable_errors()) {
			if (fix_project_errors(ar)) {
				solution_has_been_fixed = true;
				reset_root();
				continue;
			} else {
				if (!err_msg.isEmpty()) {
					QMessageBox* box = new QMessageBox(QMessageBox::Critical,
					                                   tr("Error while loading solution..."),
					                                   err_msg, QMessageBox::Ok, this);
					box->exec();
				}
				reset_root();
				return false;
			}
		}
		break;
	}

	// Update GUI on loaded sources.
	for (Inendi::PVSource* src : get_root().get_children<Inendi::PVSource>()) {
		src->process_from_source();
		source_loaded(*src);
	}

	_root->set_path(file);

	set_window_title_with_filename();
	if (solution_has_been_fixed) {
		setWindowModified(true);
	}

	PVHive::call<FUNC(PVCore::PVRecentItemsManager::add)>(
	    PVCore::PVRecentItemsManager::get(), file,
	    PVCore::PVRecentItemsManager::Category::PROJECTS);

	flag_investigation_as_cached(file);

	return true;
}

void PVInspector::PVMainWindow::save_solution(QString const& file,
                                              PVCore::PVSerializeArchiveOptions_p const& options)
{
	try {
		PVCore::PVProgressBox* pbox_solution =
		    new PVCore::PVProgressBox("Saving investigation...", this);
		pbox_solution->set_enable_cancel(true);
		bool ret = PVCore::PVProgressBox::progress([&] { get_root().save_to_file(file, options); },
		                                           pbox_solution);
		if (!ret) {
			return;
		}
	} catch (PVCore::PVSerializeArchiveError const& e) {
		QMessageBox* box =
		    new QMessageBox(QMessageBox::Critical, tr("Error while saving solution..."),
		                    tr("Error while saving solution %1:\n%2").arg(file).arg(e.what()),
		                    QMessageBox::Ok, this);
		box->exec();
	}

	PVHive::call<FUNC(PVCore::PVRecentItemsManager::add)>(
	    PVCore::PVRecentItemsManager::get(), file,
	    PVCore::PVRecentItemsManager::Category::PROJECTS);

	flag_investigation_as_cached(file);

	set_window_title_with_filename();
}

void PVInspector::PVMainWindow::flag_investigation_as_cached(const QString& investigation)
{
	QStringList nraws;
	for (Inendi::PVSource* source : get_root().get_children<Inendi::PVSource>()) {
		nraws << QString::fromStdString(source->get_rushnraw().collection().rootdir());
	}
	PVRush::PVNrawCacheManager::get().add_investigation(investigation, nraws);
}

void PVInspector::PVMainWindow::set_window_title_with_filename()
{
	static int sequenceNumber = 1;

	QString file;
	if (is_solution_untitled()) {
		file = tr("new-solution%1." INENDI_ROOT_ARCHIVE_EXT).arg(sequenceNumber++);
	} else {
		file = QFileInfo(get_solution_path()).canonicalFilePath();
	}

	setWindowModified(false);
	setWindowFilePath(file);
}

bool PVInspector::PVMainWindow::fix_project_errors(PVCore::PVSerializeArchive_p ar)
{
	// Fix errors due to invalid file paths
	PVCore::PVSerializeArchive::list_errors_t errs_file =
	    ar->get_repairable_errors_of_type<PVCore::PVSerializeArchiveErrorFileNotReadable>();
	// TODO: a nice widget were file paths can be modified by batch (for instance
	// modify all the files' directory in one action)
	foreach (PVCore::PVSerializeArchiveFixError_p err, errs_file) {
		QString const& old_path(
		    err->exception_as<PVCore::PVSerializeArchiveErrorFileNotReadable>()->get_path());
		QMessageBox* box =
		    new QMessageBox(QMessageBox::Warning, tr("Error while loading project..."),
		                    tr("File '%1' cannot be found or isn't readable by the process. Please "
		                       "select its new path.")
		                        .arg(old_path),
		                    QMessageBox::Ok, this);
		box->exec();
		QString new_file =
		    QFileDialog::getOpenFileName(this, tr("Select new file path..."), old_path);
		if (new_file.isEmpty()) {
			return false;
		}
		PVCore::PVSerializeArchiveFixAttribute* fix_a =
		    (PVCore::PVSerializeArchiveFixAttribute*)err.get();
		fix_a->fix(new_file);
	}

	// Return true if and only if all the errors have been fixed.
	return !ar->has_repairable_errors();
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
 * PVInspector::PVMainWindow::selection_inverse_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::selection_all_Slot()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);
	if (!current_view()) {
		return;
	}

	Inendi::PVView_sp lib_view(current_view()->shared_from_this());
	if (lib_view) {
		lib_view->select_all_nonzb_lines();
		PVHive::PVCallHelper::call<FUNC(Inendi::PVView::process_real_output_selection)>(lib_view);
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

	Inendi::PVView_sp lib_view(current_view()->shared_from_this());
	if (lib_view) {
		lib_view->select_no_line();
		PVHive::PVCallHelper::call<FUNC(Inendi::PVView::process_real_output_selection)>(lib_view);
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

	Inendi::PVView_sp lib_view(current_view()->shared_from_this());
	if (lib_view) {
		lib_view->select_inv_lines();
		PVHive::PVCallHelper::call<FUNC(Inendi::PVView::process_real_output_selection)>(lib_view);
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
 * PVInspector::PVMainWindow::update_reply_finished_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::update_reply_finished_Slot(QNetworkReply* reply)
{
	if (reply->error() != QNetworkReply::NoError) {
		// There was an error retrieving the current version.
		// Maybe inendi has no internet access !
		PVLOG_DEBUG("(PVMainWindow::update_reply_finished_Slot) network error\n");
		return;
	}

	QByteArray data = reply->readAll();
	version_t current_v, last_v;
	if (!PVCore::PVVersion::from_network_reply(data, current_v, last_v)) {
		PVLOG_DEBUG("(PVMainWindow::update_reply_finished_Slot) invalid server "
		            "reply:\n%s\n",
		            qPrintable(QString::fromUtf8(data.constData(), data.size())));
		return;
	}

	if (INENDI_MAJOR_VERSION(current_v) != INENDI_CURRENT_VERSION_MAJOR ||
	    INENDI_MINOR_VERSION(current_v) != INENDI_CURRENT_VERSION_MINOR ||
	    last_v < INENDI_CURRENT_VERSION) {
		// Invalid answer from the server
		PVLOG_DEBUG("(PVMainWindow::update_reply_finished_Slot) invalid server "
		            "reply: version mismatch:\ncurrent version: %s / last current "
		            "major/minor version: %s\nlast available version: %s.",
		            INENDI_CURRENT_VERSION_STR, qPrintable(PVCore::PVVersion::to_str(current_v)),
		            qPrintable(PVCore::PVVersion::to_str(last_v)));
		return;
	}

	if (current_v == _last_known_cur_release && last_v == _last_known_maj_release) {
		// We already informed the user once.
		// Display version informations
		return;
	}

	_last_known_cur_release = current_v;
	_last_known_maj_release = last_v;

	// Update PVCONFIG settings
	QSettings& pvconfig = PVCore::PVConfig::get().config();

	pvconfig.setValue(PVCONFIG_LAST_KNOWN_CUR_RELEASE, current_v);
	pvconfig.setValue(PVCONFIG_LAST_KNOWN_MAJ_RELEASE, last_v);

	QString desc = tr("Your current version is %1.\n").arg(INENDI_CURRENT_VERSION_STR);
	bool show_msg = false;
	if (current_v > INENDI_CURRENT_VERSION) {
		// A patch is available
		desc += tr("A new version (%1) is available for free for the %2.%3 branch.")
		            .arg(PVCore::PVVersion::to_str(current_v))
		            .arg(INENDI_CURRENT_VERSION_MAJOR)
		            .arg(INENDI_CURRENT_VERSION_MINOR);
		desc += "\n";
		show_msg = true;
	}
	if (last_v != current_v && last_v > INENDI_CURRENT_VERSION) {
		// A new major release is available
		desc += tr("A new major release (%1) is available.").arg(PVCore::PVVersion::to_str(last_v));
		show_msg = true;
	}

	if (show_msg) {
		PVLOG_INFO(qPrintable(desc));
		QMessageBox msgBox(QMessageBox::Information, tr("New version available"),
		                   tr("A new version is available.\n\n") + desc, QMessageBox::Ok, this);
		msgBox.exec();
	}
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
				p = PVCore::get_qobject_hierarchy_of_type<PVGuiQt::PVSceneWorkspacesTabWidget>(w);
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
			int current_tab_index =
			    _projects_tab_widget->current_workspace_tab_widget()->currentIndex();
			name = QFileInfo(_projects_tab_widget->current_workspace_tab_widget()->tabText(
			                     current_tab_index))
			           .baseName();
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

	save_screenshot(pixmap, "Save window capture", "application");
}

/******************************************************************************
 * PVInspector::PVMainWindow::get_screenshot_desktop
 *****************************************************************************/

void PVInspector::PVMainWindow::get_screenshot_desktop()
{
	QPixmap pixmap = QPixmap::grabWindow(QApplication::desktop()->winId());

	save_screenshot(pixmap, "Save desktop capture", "desktop");
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::new_format_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::new_format_Slot()
{
	PVFormatBuilderWidget* editorWidget = new PVFormatBuilderWidget(this);
	editorWidget->show();
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::cur_format_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::cur_format_Slot()
{
	if (not current_view()) {
		// FIXME : This button should not be available in this case.
		return;
	}

	Inendi::PVSource& cur_src = current_view()->get_parent<Inendi::PVSource>();
	PVRush::PVFormat const& format = cur_src.get_format();
	if (format.get_full_path().isEmpty()) {
		return;
	}

	PVFormatBuilderWidget* editorWidget =
	    new PVFormatBuilderWidget(_projects_tab_widget->current_workspace());
	editorWidget->openFormat(format.get_full_path());
	editorWidget->show();
}

void PVInspector::PVMainWindow::edit_format_Slot(const QString& format)
{
	PVFormatBuilderWidget* editorWidget =
	    new PVFormatBuilderWidget(_projects_tab_widget->current_workspace());
	editorWidget->openFormat(format);
	editorWidget->show();
}

void PVInspector::PVMainWindow::open_format_Slot()
{
	PVFormatBuilderWidget* editorWidget = new PVFormatBuilderWidget(this);
	QString url = editorWidget->slotOpen();

	if (!url.isEmpty()) {
		editorWidget->show();
		PVHive::call<FUNC(PVCore::PVRecentItemsManager::add)>(
		    PVCore::PVRecentItemsManager::get(), url,
		    PVCore::PVRecentItemsManager::Category::EDITED_FORMATS);
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::enable_menu_filter_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::enable_menu_filter_Slot(bool f)
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);
	filter_Menu->setEnabled(f);
}

void PVInspector::PVMainWindow::edit_format_Slot(QString const& path, QWidget* parent)
{
	PVFormatBuilderWidget* editorWidget = new PVFormatBuilderWidget(parent);
	editorWidget->show();
	editorWidget->openFormat(path);
}

void PVInspector::PVMainWindow::edit_format_Slot(QDomDocument& doc, QWidget* parent)
{
	PVFormatBuilderWidget* editorWidget = new PVFormatBuilderWidget(parent);
	editorWidget->show();
	editorWidget->openFormat(doc);
}

void PVInspector::PVMainWindow::selection_set_from_current_layer_Slot()
{
	if (current_view()) {
		Inendi::PVView_sp view(current_view()->shared_from_this());
		set_selection_from_layer(view, view->get_current_layer());
	}
}

void PVInspector::PVMainWindow::selection_set_from_layer_Slot()
{
	if (current_view()) {
		Inendi::PVView_sp view(current_view()->shared_from_this());

		PVCore::PVArgumentList args;
		args[PVCore::PVArgumentKey("sel-layer", tr("Choose a layer"))].setValue<Inendi::PVLayer*>(
		    &view->get_current_layer());
		bool ret = PVWidgets::PVArgumentListWidget::modify_arguments_dlg(
		    PVWidgets::PVArgumentListWidgetFactory::create_layer_widget_factory(*view), args, this);
		if (ret) {
			Inendi::PVLayer* layer = args["sel-layer"].value<Inendi::PVLayer*>();
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

void PVInspector::PVMainWindow::root_modified()
{
	setWindowModified(true);
}
