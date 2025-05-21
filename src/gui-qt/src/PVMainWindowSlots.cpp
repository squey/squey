//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/rush/PVNrawCacheManager.h>

#include <pvkernel/core/qobject_helpers.h>
#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/core/PVRecentItemsManager.h>
#include <pvkernel/core/PVArchive.h>
#include <pvkernel/core/PVSerializeArchiveZip.h>
#include <pvkernel/core/PVSerializeArchiveFixError.h>
#include <pvkernel/widgets/PVFileDialog.h>

#include <squey/widgets/editors/PVAxisIndexEditor.h>

#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>

#include <pvguiqt/PVAxesCombinationDialog.h>
#include <pvguiqt/PVImportSourceToProjectDlg.h>
#include <pvguiqt/PVWorkspace.h>
#include <pvguiqt/PVWorkspacesTabWidget.h>
#include <pvguiqt/PVExportSelectionDlg.h>
#include <pvguiqt/PVAboutBoxDialog.h>
#include <pvguiqt/PVCredentialDialog.h>

#include <PVMainWindow.h>
#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <squey/widgets/PVArgumentListWidgetFactory.h>
#include <PVFormatBuilderWidget.h>
#include <PVSaveDataTreeDialog.h>

#include <QApplication>
#include <QPainter>
#include <QDockWidget>
#include <QDesktopServices>
#include <QWindow>
#include <QScreen>

#include <boost/thread/scoped_thread.hpp>
#include <memory>

/******************************************************************************
 *
 * App::PVMainWindow::about_Slot()
 *
 *****************************************************************************/

void App::PVMainWindow::about_Slot(PVGuiQt::PVAboutBoxDialog::Tab tab)
{
	auto* about_dialog = new PVGuiQt::PVAboutBoxDialog(tab, this);
	about_dialog->exec();
	about_dialog->deleteLater();
}

void App::PVMainWindow::axes_combination_editor_Slot()
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
 * App::PVMainWindow::commit_selection_to_new_layer_Slot
 *
 *****************************************************************************/
void App::PVMainWindow::commit_selection_to_new_layer_Slot()
{
	if (!current_view()) {
		return;
	}

	Squey::PVView* current_lib_view = current_view();
	commit_selection_to_new_layer(current_lib_view);
}

void App::PVMainWindow::move_selection_to_new_layer_Slot()
{
	if (!current_view()) {
		return;
	}

	Squey::PVView* current_lib_view = current_view();
	move_selection_to_new_layer(current_lib_view);
}

/******************************************************************************
 *
 * App::PVMainWindow::events_display_unselected_zombies_parallelview_Slot()
 *
 *****************************************************************************/
void App::PVMainWindow::events_display_unselected_zombies_parallelview_Slot()
{
	if (!current_view()) {
		return;
	}

	/* We refresh the listing */
	current_view()->toggle_view_unselected_zombie_visibility();
}

/******************************************************************************
 *
 * App::PVMainWindow::export_selection_Slot
 *
 *****************************************************************************/
void App::PVMainWindow::export_selection_Slot()
{
	PVLOG_DEBUG("App::PVMainWindow::%s\n", __FUNCTION__);

	Squey::PVView* view = current_view();
	Squey::PVSelection const& sel = view->get_real_output_selection();

	PVGuiQt::PVExportSelectionDlg::export_selection(*view, sel, this);
}

bool App::PVMainWindow::load_source_from_description_Slot(
    PVRush::PVSourceDescription src_desc)
{
	bool has_error = false;
	Squey::PVScene* scene_p;

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
		scene_p = &_projects_tab_widget->project_new();
		new_scene = true;
	} else if (scenes.size() == 1) {
		// Only one project loaded: use it to load the source
		scene_p = scenes.front();
		get_root().select_scene(*scene_p);
	} else {
		// More than one project loaded: ask the user the project he wants to use to
		// load the source
		auto* dlg =
		    new PVGuiQt::PVImportSourceToProjectDlg(get_root(), get_root().current_scene(), this);
		if (dlg->exec() != QDialog::Accepted) {
			return false;
		}

		get_root().select_scene(*((Squey::PVScene*)dlg->get_selected_scene()));
		scene_p = current_scene();
		dlg->deleteLater();
	}

	Squey::PVSource* src_p = nullptr;
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

void App::PVMainWindow::solution_new_Slot()
{
	// FIXME : This Windows is a memory leak
	auto* new_mw = new PVMainWindow();
	new_mw->show();
	new_mw->set_window_title_with_filename();
}

void App::PVMainWindow::solution_load_Slot()
{
	_load_solution_dlg.setFileMode(QFileDialog::ExistingFile);
	_load_solution_dlg.setAcceptMode(QFileDialog::AcceptOpen);
	if (_load_solution_dlg.exec() != QDialog::Accepted) {
		return;
	}
	QString file = _load_solution_dlg.selectedFiles().at(0);
	load_solution_and_create_mw(file);
}

void App::PVMainWindow::load_solution_and_create_mw(QString const& file)
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
		auto* other = new PVMainWindow();
		other->show();
		if (!other->load_solution(file)) {
			other->deleteLater();
			return;
		}
	}
}

void App::PVMainWindow::solution_save_Slot()
{
	if (is_solution_untitled()) {
		solution_saveas_Slot();
	} else {
		// FIXME : We should remember if log_file have to be saved
		save_solution(get_solution_path());
	}
}

void App::PVMainWindow::solution_saveas_Slot()
{
	if (get_root().is_empty()) {
		return;
	}

	PVSaveDataTreeDialog dlg(SQUEY_ROOT_ARCHIVE_EXT, SQUEY_ROOT_ARCHIVE_FILTER, this);
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

bool App::PVMainWindow::maybe_save_solution()
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

bool App::PVMainWindow::load_solution(QString const& file)
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
			        ar = std::make_unique<PVCore::PVSerializeArchiveZip>(
			            file, PVCore::PVSerializeArchive::read, SQUEY_ARCHIVES_VERSION);
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
#if 0 // Disable this for now as an investigation doesn't need to reuse credentials
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
#endif
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
	_projects_tab_widget->increase_sequence(get_root().size());

	// Update GUI on loaded sources.
	for (Squey::PVSource* src : get_root().get_children<Squey::PVSource>()) {
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

void App::PVMainWindow::save_solution(QString const& file, bool save_log_file)
{
	if (PVCore::PVProgressBox::progress(
	        [&](PVCore::PVProgressBox& pbox) {
		        pbox.set_enable_cancel(true);
		        pbox.set_extended_status("Opening investigation for saving...");
		        PVCore::PVSerializeArchiveZip ar(file, PVCore::PVSerializeArchive::write,
		                                         SQUEY_ARCHIVES_VERSION, save_log_file);
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

void App::PVMainWindow::flag_investigation_as_cached(const QString& investigation)
{
	QStringList nraws;
	for (Squey::PVSource* source : get_root().get_children<Squey::PVSource>()) {
		nraws << QString::fromStdString(source->get_rushnraw().dir());
	}
	PVRush::PVNrawCacheManager::get().add_investigation(investigation, nraws);
}

void App::PVMainWindow::set_window_title_with_filename()
{
	QString file;
	if (is_solution_untitled()) {
		static int sequenceNumber = 1;
		file = tr("new-solution%1." SQUEY_ROOT_ARCHIVE_EXT).arg(sequenceNumber++);
	} else {
		file = QFileInfo(get_solution_path()).canonicalFilePath();
		tools_cur_format_Action->setEnabled(false);
	}

	pvlogger::info() << "file=" << qPrintable(file) << std::endl;

	setWindowModified(false);
	setWindowFilePath(file);
}

/******************************************************************************
 *
 * App::PVMainWindow::quit_Slot
 *
 *****************************************************************************/
void App::PVMainWindow::quit_Slot()
{
	close();
}

/******************************************************************************
 *
 * App::PVMainWindow::selection_inverse_Slot()
 *
 *****************************************************************************/
void App::PVMainWindow::selection_all_Slot()
{
	PVLOG_DEBUG("App::PVMainWindow::%s\n", __FUNCTION__);
	if (!current_view()) {
		return;
	}

	current_view()->select_all();
}

/******************************************************************************
 *
 * App::PVMainWindow::selection_inverse_Slot()
 *
 *****************************************************************************/
void App::PVMainWindow::selection_none_Slot()
{
	PVLOG_DEBUG("App::PVMainWindow::%s\n", __FUNCTION__);
	if (!current_view()) {
		return;
	}

	current_view()->select_none();
}

/******************************************************************************
 *
 * App::PVMainWindow::selection_inverse_Slot()
 *
 *****************************************************************************/
void App::PVMainWindow::selection_inverse_Slot()
{
	PVLOG_DEBUG("App::PVMainWindow::%s\n", __FUNCTION__);
	if (!current_view()) {
		return;
	}

	current_view()->select_inverse();
}

/******************************************************************************
 *
 * App::PVMainWindow::set_color_Slot()
 *
 *****************************************************************************/
void App::PVMainWindow::set_color_Slot()
{
	PVLOG_DEBUG("App::PVMainWindow::%s\n", __FUNCTION__);

	/* CODE */
	if (!current_view())
		return;
	set_color(current_view());
}

/******************************************************************************
 * App::PVMainWindow::get_screenshot_widget
 *****************************************************************************/

void App::PVMainWindow::get_screenshot_widget()
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
			    _projects_tab_widget->current_workspace_tab_widget()->current_index();
			name = QFileInfo(_projects_tab_widget->current_workspace_tab_widget()->tab_text(
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

QScreen* App::PVMainWindow::get_screen() const
{
	QScreen* screen = QGuiApplication::primaryScreen();
	if (const QWindow* window = QWidget::windowHandle()) {
		screen = window->screen();
	}
	assert(screen);

	return screen;
}

/******************************************************************************
 * App::PVMainWindow::get_screenshot_window
 *****************************************************************************/

void App::PVMainWindow::get_screenshot_window()
{
	QPixmap pixmap = get_screen()->grabWindow(winId());

	save_screenshot(pixmap, "Save window capture", "application");
}

/******************************************************************************
 * App::PVMainWindow::get_screenshot_desktop
 *****************************************************************************/

void App::PVMainWindow::get_screenshot_desktop()
{
	QPixmap pixmap = get_screen()->grabWindow(0);

	save_screenshot(pixmap, "Save desktop capture", "desktop");
}

/******************************************************************************
 *
 * App::PVMainWindow::new_format_Slot()
 *
 *****************************************************************************/
void App::PVMainWindow::new_format_Slot()
{
	auto* editorWidget = new PVFormatBuilderWidget(this);
	editorWidget->show();
}

/******************************************************************************
 *
 * App::PVMainWindow::cur_format_Slot()
 *
 *****************************************************************************/
void App::PVMainWindow::cur_format_Slot()
{
	if (not current_view()) {
		// FIXME : This button should not be available in this case.
		return;
	}

	auto& cur_src = current_view()->get_parent<Squey::PVSource>();
	PVRush::PVFormat const& format = cur_src.get_original_format();
	if (format.get_full_path().isEmpty()) {
		return;
	}

	auto* editorWidget =
	    new PVFormatBuilderWidget(_projects_tab_widget->current_workspace());
	editorWidget->openFormat(format.get_full_path());
	editorWidget->show();
}

void App::PVMainWindow::edit_format_Slot(const QString& format)
{
	auto* editorWidget =
	    new PVFormatBuilderWidget(_projects_tab_widget->current_workspace());
	editorWidget->openFormat(format);
	editorWidget->show();
}

void App::PVMainWindow::open_format_Slot()
{
	auto* editorWidget = new PVFormatBuilderWidget(_projects_tab_widget->current_workspace());
	QString url = editorWidget->slotOpen();

	if (!url.isEmpty()) {
		editorWidget->show();
		PVCore::PVRecentItemsManager::get().add<PVCore::Category::EDITED_FORMATS>(url);
	}
}

void App::PVMainWindow::edit_format_Slot(QString const& path, QWidget* parent)
{
	auto* editorWidget = new PVFormatBuilderWidget(parent);
	editorWidget->show();
	editorWidget->openFormat(path);
}

void App::PVMainWindow::edit_format_Slot(QDomDocument& doc, QWidget* parent)
{
	auto* editorWidget = new PVFormatBuilderWidget(parent);
	editorWidget->show();
	editorWidget->openFormat(doc);
}

void App::PVMainWindow::selection_set_from_current_layer_Slot()
{
	if (current_view()) {
		current_view()->set_selection_from_layer(current_view()->get_current_layer());
	}
}

void App::PVMainWindow::selection_set_from_layer_Slot()
{
	if (current_view()) {
		PVCore::PVArgumentList args;
		args[PVCore::PVArgumentKey("sel-layer", tr("Choose a layer"))].setValue<Squey::PVLayer*>(
		    &current_view()->get_current_layer());
		bool ret = PVWidgets::PVArgumentListWidget::modify_arguments_dlg(
		    PVWidgets::PVArgumentListWidgetFactory::create_layer_widget_factory(*current_view()),
		    args, this);
		if (ret) {
			auto* layer = args["sel-layer"].value<Squey::PVLayer*>();
			current_view()->set_selection_from_layer(*layer);
		}
	}
}

void App::PVMainWindow::view_display_inv_elts_Slot()
{
	if (current_view()) {
		display_inv_elts();
	}
}

void App::PVMainWindow::root_modified()
{
	setWindowModified(true);
}
