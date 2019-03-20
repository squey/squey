/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <pvkernel/rush/PVNrawCacheManager.h>

#include <pvkernel/core/qobject_helpers.h>
#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/core/PVRecentItemsManager.h>
#include <pvkernel/core/PVArchive.h>
#include <pvkernel/core/PVSerializeArchiveZip.h>
#include <pvkernel/core/PVSerializeArchiveFixError.h>
#include <pvkernel/widgets/PVFileDialog.h>

#ifdef WITH_MINESET
#include <inendi/PVMineset.h>
#endif

#include <inendi/widgets/editors/PVAxisIndexEditor.h>

#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>

#include <pvguiqt/PVAxesCombinationDialog.h>
#include <pvguiqt/PVLayerFilterProcessWidget.h>
#include <pvguiqt/PVImportSourceToProjectDlg.h>
#include <pvguiqt/PVWorkspace.h>
#include <pvguiqt/PVWorkspacesTabWidget.h>
#include <pvguiqt/PVExportSelectionDlg.h>
#include <pvguiqt/PVAboutBoxDialog.h>
#include <pvguiqt/PVCredentialDialog.h>

#include <PVMainWindow.h>
#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <inendi/widgets/PVArgumentListWidgetFactory.h>
#include <PVFormatBuilderWidget.h>
#include <PVSaveDataTreeDialog.h>

#include <QPainter>
#include <QDockWidget>
#include <QDesktopServices>
#include <QDesktopWidget>
#include <QWindow>
#include <QScreen>

#include <boost/thread/scoped_thread.hpp>

/******************************************************************************
 *
 * PVInspector::PVMainWindow::about_Slot()
 *
 *****************************************************************************/

int PVInspector::PVMainWindow::sequence_n = 1;

void PVInspector::PVMainWindow::about_Slot(PVGuiQt::PVAboutBoxDialog::Tab tab)
{
	PVGuiQt::PVAboutBoxDialog* about_dialog = new PVGuiQt::PVAboutBoxDialog(tab, this);
	if (tab == PVGuiQt::PVAboutBoxDialog::Tab::REFERENCE_MANUAL) {
		about_dialog->resize(1550, 950);
	}
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
	if (!current_view()) {
		return;
	}

	current_view()->toggle_listing_unselected_visibility();
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::events_display_zombies_listing_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::events_display_zombies_listing_Slot()
{
	if (!current_view()) {
		return;
	}

	current_view()->toggle_listing_zombie_visibility();
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::events_display_unselected_zombies_parallelview_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::events_display_unselected_zombies_parallelview_Slot()
{
	if (!current_view()) {
		return;
	}

	/* We refresh the listing */
	current_view()->toggle_view_unselected_zombie_visibility();
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

	PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& pbox) {
		    pbox.set_enable_cancel(false);
		    try {
			    std::string dataset_url = Inendi::PVMineset::import_dataset(*current_view());
			    current_view()->add_mineset_dataset(dataset_url);
			    QDesktopServices::openUrl(QUrl(dataset_url.c_str()));
		    } catch (const Inendi::PVMineset::mineset_error& e) {
			    pbox.critical("Error when exporting current selection to Mineset", e.what());
		    }
		},
	    "Exporting data to Mineset...", this);
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
	Inendi::PVScene& scene_p = get_root().emplace_add_child(get_next_scene_name());
	_projects_tab_widget->add_project(scene_p);

	return scene_p;
}

bool PVInspector::PVMainWindow::load_source_from_description_Slot(
    PVRush::PVSourceDescription src_desc)
{
	bool has_error = false;
	Inendi::PVScene* scene_p;

	PVRush::PVFormat format = src_desc.get_format();

	const size_t axes_count = format.get_axes().size();

	if (axes_count < 2) {
		const QString text((axes_count == 0) ? "no axis" : "only one axis");

		QMessageBox::critical(
		    this, tr("Format \"%1\" is invalid").arg(format.get_full_path()),
		    tr("It has %2, it must define at least 2 axes to be usable.").arg(text));

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
		get_root().select_scene(*scene_p);
	} else {
		// More than one project loaded: ask the user the project he wants to use to
		// load the source
		PVGuiQt::PVImportSourceToProjectDlg* dlg =
		    new PVGuiQt::PVImportSourceToProjectDlg(get_root(), get_root().current_scene(), this);
		if (dlg->exec() != QDialog::Accepted) {
			return false;
		}

		get_root().select_scene(*((Inendi::PVScene*)dlg->get_selected_scene()));
		scene_p = current_scene();
		dlg->deleteLater();
	}

	Inendi::PVSource* src_p = nullptr;
	try {
		src_p = &scene_p->emplace_add_child(src_desc);
	} catch (PVRush::PVFormatException const& e) {
		QMessageBox::critical(this, "Error with format...", e.what());
		has_error = true;
	} catch (PVFilter::PVFieldsFilterInvalidArguments const& e) {
		QMessageBox::critical(this, "Error", e.what());
		has_error = true;
	} catch (PVRush::PVInputException const& e) {
		QMessageBox::critical(this, tr("Fatal error while loading source..."),
		                      tr("Fatal error while loading source: %1").arg(e.what()));
		has_error = true;
	}

	if (has_error) {
		if (new_scene) {
			_projects_tab_widget->remove_project(
			    _projects_tab_widget->get_workspace_tab_widget_from_scene(scene_p));
		}
		return false;
	}

	try {
		if (not load_source(src_p)) {
			remove_source(src_p);
			return false;
		}
	} catch (const PVRush::PVFormatInvalidTime& e) {
		QMessageBox::critical(this, tr("Fatal error while loading source..."), e.what());
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
		// FIXME : We should remember if log_file have to be saved
		save_solution(get_solution_path());
	}
}

void PVInspector::PVMainWindow::solution_saveas_Slot()
{
	if (get_root().is_empty()) {
		return;
	}

	PVSaveDataTreeDialog dlg(INENDI_ROOT_ARCHIVE_EXT, INENDI_ROOT_ARCHIVE_FILTER, this);
	if (!_current_save_root_folder.isEmpty()) {
		dlg.setDirectory(_current_save_root_folder);
	}
	dlg.selectFile(get_solution_path());
	if (dlg.exec() == QDialog::Accepted) {
		QString file = dlg.selectedFiles().at(0);
		save_solution(file, dlg.save_log_file());
	}
	_current_save_root_folder = dlg.directory().absolutePath();
}

bool PVInspector::PVMainWindow::maybe_save_solution()
{
	if (isWindowModified()) {
		QMessageBox::StandardButton ret;
		QString solution_name = QFileInfo(windowFilePath()).fileName();
		ret = QMessageBox::warning(this, tr("%1").arg(solution_name),
		                           tr("The investigation \"%1\"has been modified.\n"
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

	PVCore::PVSerializeArchiveError read_exception("");
	bool solution_has_been_fixed = false;
	if (PVCore::PVProgressBox::progress(
	        [&](PVCore::PVProgressBox& pbox) {
		        pbox.set_enable_cancel(true);
		        pbox.set_extended_status("Opening investigation for loading...");
		        std::unique_ptr<PVCore::PVSerializeArchive> ar;
		        try {
			        ar.reset(new PVCore::PVSerializeArchiveZip(
			            file, PVCore::PVSerializeArchive::read, INENDI_ARCHIVES_VERSION));
		        } catch (const PVCore::PVSerializeArchiveError& e) {
			        read_exception = e;
			        return;
		        }

		        // Use a scoped thread as it will continue forever so we want to interrupt is and
		        // abort at then end.
		        // This thread update the progressBox status every 100 ms
		        boost::strict_scoped_thread<boost::interrupt_and_join_if_joinable> t1(
		            (boost::thread([&ar, &pbox]() {
			            while (true) {
				            pbox.set_extended_status(ar->get_current_status());
				            boost::this_thread::interruption_point();
				            std::this_thread::sleep_for(std::chrono::milliseconds(100));
			            }
			        })));

		        while (true) {
			        QString err_msg;
			        try {
				        pbox.set_extended_status("Loading investigation...");
				        get_root().load_from_archive(*ar);
			        } catch (PVCore::PVSerializeArchiveError& e) {
				        read_exception = e;
				        return;
			        } catch (PVRush::PVInputException const& e) {
				        read_exception = PVCore::PVSerializeArchiveError(
				            tr("Error while loading investigation \"%1\":\n%2")
				                .arg(file)
				                .arg(e.what())
				                .toStdString());
				        return;
			        } catch (PVCore::PVSerializeReparaibleFileError const& e) {
				        pbox.warning(tr("Error while loading project %1:\n").arg(file), e.what());
				        QString old_path = QString::fromStdString(e.old_value());
				        QString new_file;

				        pbox.exec_gui([&]() {
					        new_file = PVWidgets::PVFileDialog::getOpenFileName(
					            this, tr("Select new file path..."), old_path);
					    });

				        if (new_file.isEmpty()) {
					        read_exception = PVCore::PVSerializeArchiveError(
					            tr("Error while loading investigation \"%1\":\n files can't be "
					               "found")
					                .arg(file)
					                .toStdString());
					        return;
				        }
				        // FIXME : We should be able to handle more than one error
				        ar->set_repaired_value(e.logical_path(), new_file.toStdString());
				        solution_has_been_fixed = true;
				        reset_root();
				        continue;
			        } catch (PVCore::PVSerializeReparaibleCredentialError const& e) {
				        QString login, password;
				        bool ok;
				        pbox.exec_gui([&]() {
					        PVGuiQt::CredentialDialog dial;
					        ok = (dial.exec() == QDialog::Accepted);
					        login = dial.get_login();
					        password = dial.get_password();
					    });

				        if (not ok) {
					        read_exception = PVCore::PVSerializeArchiveError(
					            "No credential provided to open this investigation");
					        return;
				        }

				        ar->set_repaired_value(e.logical_path(),
				                               (login + ";" + password).toStdString());
				        solution_has_been_fixed = true;
				        reset_root();
				        continue;
			        } catch (...) {
				        read_exception = PVCore::PVSerializeArchiveError(
				            tr("Error while loading investigation \"%1\":\n unhandled error(s).")
				                .arg(file)
				                .toStdString());
				        return;
			        }
			        break;
		        }
		    },
	        "Loading investigation...", this) != PVCore::PVProgressBox::CancelState::CONTINUE) {
		reset_root();
		return false;
	}

	if (not std::string(read_exception.what()).empty()) {
		QMessageBox::critical(this, tr("Fatal error while loading investigation..."),
		                      tr("Fatal error while loading investigation \"%1\":\n%2")
		                          .arg(file)
		                          .arg(QString::fromStdString(read_exception.what())));
		reset_root();
		return false;
	}

	// Increase counter so that later imported source have a more "distinct" name.
	// eg : We load "Data collection 1" and "Data collection 2", next imported scene
	// will be "Data collection 3". We don't care about imported invevtigation then
	// imported scene as the create a new MainWindows
	sequence_n += get_root().size();

	// Update GUI on loaded sources.
	for (Inendi::PVSource* src : get_root().get_children<Inendi::PVSource>()) {
		source_loaded(*src, false /* update_recent_items */);
	}

	_root.set_path(file);

	set_window_title_with_filename();
	if (solution_has_been_fixed) {
		setWindowModified(true);
	}

	PVCore::PVRecentItemsManager::get().add<PVCore::Category::PROJECTS>(file);

	flag_investigation_as_cached(file);

	return true;
}

void PVInspector::PVMainWindow::save_solution(QString const& file, bool save_log_file)
{
	if (PVCore::PVProgressBox::progress(
	        [&](PVCore::PVProgressBox& pbox) {
		        pbox.set_enable_cancel(true);
		        pbox.set_extended_status("Opening investigation for saving...");
		        PVCore::PVSerializeArchiveZip ar(file, PVCore::PVSerializeArchive::write,
		                                         INENDI_ARCHIVES_VERSION, save_log_file);
		        // FIXME : We should inform we are creating the zip file using RAII like scoped
		        // thread and Zip archive.

		        // Use a scoped thread as it will continue forever so we want to interrupt is
		        // and
		        // abort at then end.
		        // This thread update the progressBox status every 100 ms
		        boost::strict_scoped_thread<boost::interrupt_and_join_if_joinable> t1(
		            (boost::thread([&ar, &pbox]() {
			            while (true) {
				            pbox.set_extended_status(ar.get_current_status());
				            boost::this_thread::interruption_point();
				            std::this_thread::sleep_for(std::chrono::milliseconds(100));
			            }
			        })));

		        get_root().set_path(file);
		        get_root().save_to_file(ar);
		        try {
			        ar.close_zip();
		        } catch (PVCore::ArchiveCreationFail const& e) {
			        pbox.critical("Error while saving investigation...",
			                      "Error while saving investigation \"" + file + "\":\n" +
			                          QString::fromStdString(e.what()));
			        pbox.set_canceled();
		        } catch (PVCore::PVSerializeArchiveError const& e) {
			        pbox.critical("Error while saving investigation...",
			                      "Error while saving investigation \"" + file + "\":\n" +
			                          QString::fromStdString(e.what()));
			        pbox.set_canceled();
		        }
		    },
	        "Saving investigation...", this) != PVCore::PVProgressBox::CancelState::CONTINUE) {
		return;
	}

	PVCore::PVRecentItemsManager::get().add<PVCore::Category::PROJECTS>(file);

	flag_investigation_as_cached(file);

	set_window_title_with_filename();
}

void PVInspector::PVMainWindow::flag_investigation_as_cached(const QString& investigation)
{
	QStringList nraws;
	for (Inendi::PVSource* source : get_root().get_children<Inendi::PVSource>()) {
		nraws << QString::fromStdString(source->get_rushnraw().dir());
	}
	PVRush::PVNrawCacheManager::get().add_investigation(investigation, nraws);
}

void PVInspector::PVMainWindow::set_window_title_with_filename()
{
	QString file;
	if (is_solution_untitled()) {
		static int sequenceNumber = 1;
		file = tr("new-solution%1." INENDI_ROOT_ARCHIVE_EXT).arg(sequenceNumber++);
	} else {
		file = QFileInfo(get_solution_path()).canonicalFilePath();
		tools_cur_format_Action->setEnabled(false);
	}

	setWindowModified(false);
	setWindowFilePath(file);
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

	current_view()->select_all();
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

	current_view()->select_none();
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

	current_view()->select_inverse();
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

	QPixmap pixmap = p->grab();

	save_screenshot(pixmap, "Save view capture", name);
}

QScreen* PVInspector::PVMainWindow::get_screen() const
{
	QScreen* screen = QGuiApplication::primaryScreen();
	if (const QWindow* window = QWidget::windowHandle()) {
		screen = window->screen();
	}
	assert(screen);

	return screen;
}

/******************************************************************************
 * PVInspector::PVMainWindow::get_screenshot_window
 *****************************************************************************/

void PVInspector::PVMainWindow::get_screenshot_window()
{
	QPixmap pixmap = get_screen()->grabWindow(winId());

	save_screenshot(pixmap, "Save window capture", "application");
}

/******************************************************************************
 * PVInspector::PVMainWindow::get_screenshot_desktop
 *****************************************************************************/

void PVInspector::PVMainWindow::get_screenshot_desktop()
{
	QPixmap pixmap = get_screen()->grabWindow(QApplication::desktop()->winId());

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
		PVCore::PVRecentItemsManager::get().add<PVCore::Category::EDITED_FORMATS>(url);
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
		current_view()->set_selection_from_layer(current_view()->get_current_layer());
	}
}

void PVInspector::PVMainWindow::selection_set_from_layer_Slot()
{
	if (current_view()) {
		PVCore::PVArgumentList args;
		args[PVCore::PVArgumentKey("sel-layer", tr("Choose a layer"))].setValue<Inendi::PVLayer*>(
		    &current_view()->get_current_layer());
		bool ret = PVWidgets::PVArgumentListWidget::modify_arguments_dlg(
		    PVWidgets::PVArgumentListWidgetFactory::create_layer_widget_factory(*current_view()),
		    args, this);
		if (ret) {
			Inendi::PVLayer* layer = args["sel-layer"].value<Inendi::PVLayer*>();
			current_view()->set_selection_from_layer(*layer);
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
