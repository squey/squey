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

#include <QApplication>
#include <QDialog>
#include <QFile>
#include <QFontDatabase>
#include <QLabel>
#include <QMessageBox>
#include <QMimeData>
#include <QRegExp>
#include <QShortcut>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QStyleHints>
#include <QStyleFactory>
#include <QTimer>

#include <PVImportController.h>
#include <PVMainWindow.h>
#include <PVStringListChooserWidget.h>

#include <pvguiqt/PVGuiQtSettings.h>
#include <pvguiqt/PVImportPage.h>
#include <pvkernel/core/PVTaskProgress.h>
#include <pvkernel/widgets/PVImportWidget.h>
#include <pvkernel/core/PVKernelSettings.h>
#include <pvguiqt/PVWorkspace.h>
#include <squey/PVAnalysisHistory.h>
#include <squey/widgets/PVNewLayerDialog.h>

#include <pvkernel/core/PVRecentItemsManager.h>
#include <pvbase/general.h>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <pvkernel/core/PVConfig.h>
#include <pvkernel/core/PVFlatpakHelper.h>

#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVMeanValue.h>
#include <pvkernel/core/PVProgressBox.h>

#include <pvkernel/rush/PVFileDescription.h>

#include <pvkernel/widgets/PVColorDialog.h>
#include <pvkernel/widgets/PVFileDialog.h>

#include <squey/PVScaled.h>
#include <squey/PVSelection.h>
#include <squey/PVStateMachine.h>
#include <squey/PVSource.h>
#include <squey/PVAxis.h>

#include <pvparallelview/PVParallelView.h>
#include <pvguiqt/PVExportSelectionDlg.h>
#ifdef PYTHON_SUPPORT
#include <pvguiqt/PVProgressBoxPython.h>
#endif
#include <pvguiqt/PVStatusBar.h>

#include <tbb/tick_count.h>

#include <chrono>

//#include <sys/utsname.h> // uname

/******************************************************************************
 *
 * App::PVMainWindow::PVMainWindow
 *
 *****************************************************************************/
App::PVMainWindow::PVMainWindow(QWidget* parent) : QMainWindow(parent), _root()
{
	setAcceptDrops(true);

	reset_root();

	// OBJECTNAME STUFF
	setObjectName("PVMainWindow");
	QGuiApplication::setDesktopFileName("org.squey.Squey");

	// FIXME: check fonts licenses to be sure we can distribute them with Squey
	QFontDatabase::addApplicationFont(QString(":/Convergence-Regular.ttf"));
	QFontDatabase::addApplicationFont(QString(":/Jura-DemiBold.ttf"));
	QFontDatabase::addApplicationFont(QString(":/OSP-DIN.ttf"));
	QFontDatabase::addApplicationFont(QString(":/PT_Sans-Narrow-Web-Bold.ttf"));
	QFontDatabase::addApplicationFont(QString(":/PT_Sans-Narrow-Web-Regular.ttf"));
	QFontDatabase::addApplicationFont(QString(":/BlackHanSans-Regular.ttf"));

	// import_source = nullptr;
	report_started = false;
	report_image_index = 0;
	report_filename = nullptr;

	// We activate all available Windows
	_projects_tab_widget = new PVGuiQt::PVProjectsTabWidget(&get_root());
	_import_controller = std::make_unique<PVImportController>(*this);
	_projects_tab_widget->show();
	connect(_projects_tab_widget, &PVGuiQt::PVProjectsTabWidget::new_project, this,
	        &PVMainWindow::solution_new_Slot);
	connect(_projects_tab_widget, &PVGuiQt::PVProjectsTabWidget::load_project, this,
	        &PVMainWindow::solution_load_Slot);
	connect(_projects_tab_widget, &PVGuiQt::PVProjectsTabWidget::load_project_from_path, this,
	        &PVMainWindow::load_solution_and_create_mw);
	connect(_projects_tab_widget, &PVGuiQt::PVProjectsTabWidget::save_project, this,
	        &PVMainWindow::solution_save_Slot);
	connect(_projects_tab_widget, &PVGuiQt::PVProjectsTabWidget::load_source_from_description, this,
	        &PVMainWindow::load_source_from_description_Slot);
	connect(_projects_tab_widget, SIGNAL(import_type(const QString&)), this,
	        SLOT(import_type_Slot(const QString&)));
	connect(_projects_tab_widget, &PVGuiQt::PVProjectsTabWidget::new_format, this,
	        &PVMainWindow::new_format_Slot);
	connect(_projects_tab_widget, &PVGuiQt::PVProjectsTabWidget::load_format, this,
	        &PVMainWindow::open_format_Slot);
	connect(_projects_tab_widget, SIGNAL(edit_format(const QString&)), this,
	        SLOT(edit_format_Slot(const QString&)));
	connect(_projects_tab_widget, &PVGuiQt::PVProjectsTabWidget::is_empty, this,
	        &PVMainWindow::close_solution_Slot);
	connect(_projects_tab_widget, &PVGuiQt::PVProjectsTabWidget::active_project, this,
	        &PVMainWindow::menu_activate_is_file_opened);
#ifdef __linux__
	connect(&_dbus_connection, &PVCore::PVDBusConnection::import_signal,
	        [this](const QString& input_type, const QString& params_json) {
		        try {
			        _import_controller->import_params(
			            LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name(input_type),
			            params_json);
		        } catch (const std::exception& e) {
			        _projects_tab_widget->import_page().add_message(
			            PVGuiQt::PVImportPage::Severity::critical,
			            QString::fromStdString(e.what()).toHtmlEscaped());
			        _projects_tab_widget->import_page().set_closable(true);
			        _projects_tab_widget->show_import_page();
		        }
	        });
#endif

	// We display the PV Icon together with a button to import files
	pv_centralMainWidget = new QWidget();
	pv_centralMainWidget->setObjectName("pv_centralMainWidget_of_PVMainWindow");

	pv_mainLayout = new QVBoxLayout();
	pv_mainLayout->setContentsMargins(0, 0, 0, 0);
	pv_mainLayout->addWidget(_projects_tab_widget);

	// Set status bar
	PVGuiQt::PVStatusBar* status_bar = new PVGuiQt::PVStatusBar(this);
	setStatusBar(status_bar);
	statusBar()->showMessage(tr("Ready"));

	pv_centralMainWidget->setLayout(pv_mainLayout);

	pv_centralWidget = new QStackedWidget();
	pv_centralWidget->addWidget(pv_centralMainWidget);
	pv_centralWidget->setCurrentWidget(pv_centralMainWidget);

	setCentralWidget(pv_centralWidget);

	_projects_tab_widget->setFocus(Qt::OtherFocusReason);

	// We populate all actions, menus and connect them
	create_actions();
	create_menus();
	connect_actions();
	menu_activate_is_file_opened(false);

	// CSS stylesheet hot reloading
	QShortcut* refresh_theme = new QShortcut(QKeySequence(Qt::Key_Dollar), this);
	QShortcut* switch_theme = new QShortcut(QKeySequence("Shift+$"), this);
	refresh_theme->setContext(Qt::ApplicationShortcut);
	switch_theme->setContext(Qt::ApplicationShortcut);
	auto refresh_f = [&](bool switch_theme) {
		PVLOG_INFO("Reloading CSS\n");
		if (switch_theme) {
			PVCore::settings::color_scheme.set_value(
			    PVCore::PVTheme::is_color_scheme_dark() ? "light" : "dark");
		}
		else { // force refresh
			PVCore::PVTheme::set_color_scheme(PVCore::PVTheme::color_scheme());
		}
	};
	connect(refresh_theme, &QShortcut::activated, [refresh_f](){
		refresh_f(false);
	});
	connect(switch_theme, &QShortcut::activated, [refresh_f](){
		refresh_f(true);
	});

	QShortcut* toggle_zombies_shortcut = new QShortcut(QKeySequence(Qt::Key_U), this);
	QObject::connect(toggle_zombies_shortcut, &QShortcut::activated, this, &App::PVMainWindow::events_display_unselected_zombies_parallelview_Slot);

	// Last, so that the start screen is there to hear about it: adding the entry
	// is what refreshes the list it appears in.
	register_sample_dataset();
}

App::PVMainWindow::~PVMainWindow()
{
	// An import that runs reads and fills a source of _root, from threads of its own: it stops,
	// and waits for them, before clear() below drops the source.
	_import_controller.reset();

	// View widgets (workspaces, PVViewDisplay) and their Qt models
	// (PVLayerStackModel...) hold references to the PVView objects owned by
	// _root. As a value member, _root is destroyed BEFORE the child widgets
	// (destroyed by the base ~QWidget), so those widgets would outlive the views
	// they render and dereference freed memory.
	//
	// Tear the model down explicitly rather than the widget tree: every PVView
	// emits _about_to_be_delete, which is what makes the parallel views cancel
	// and drain their in-flight renderings before synchronously deleting their
	// own widget. Deleting the widgets directly skips that protocol -- a
	// background BCI rendering stays queued on the pipeline and dereferences
	// freed state from its own thread, which is how this crashed on macOS.
	get_root().clear();
}

bool App::PVMainWindow::event(QEvent* event)
{
	QString mime_type = "application/x-squey_workspace";
	// Files, which open as a command line opens them, but over what the import page asks for or
	// imports: the page takes the files its content takes itself.
	const auto takes_files = [this](QMimeData const* mime) {
		const PVGuiQt::PVImportPage& page = _projects_tab_widget->import_page();
		return not _import_controller->busy() and
		       not(page.isVisible() and page.content() != nullptr) and
		       not PVGuiQt::PVImportPage::dropped_files(mime).isEmpty();
	};

	if (event->type() == QEvent::DragEnter) {
		auto* dragEnterEvent = static_cast<QDragEnterEvent*>(event);
		if (dragEnterEvent->mimeData()->hasFormat(mime_type)) {
			dragEnterEvent->accept(); // dragEnterEvent->acceptProposedAction();
			return true;
		}
		if (takes_files(dragEnterEvent->mimeData())) {
			dragEnterEvent->acceptProposedAction();
			return true;
		}
	} else if (event->type() == QEvent::Drop) {
		auto* dropEvent = static_cast<QDropEvent*>(event);
		if (takes_files(dropEvent->mimeData())) {
			dropEvent->acceptProposedAction();
			// Once the drop is over: what the files are may need a question.
			QTimer::singleShot(0, this,
			                   [this, files = PVGuiQt::PVImportPage::dropped_files(
			                              dropEvent->mimeData())]() { load_files(files); });
			return true;
		}
		if (dropEvent->mimeData()->hasFormat(mime_type)) {
			dropEvent->acceptProposedAction();
			const QMimeData* mimeData = dropEvent->mimeData();
			QByteArray byte_array = mimeData->data(mime_type);
			if (byte_array.size() < (int)sizeof(PVGuiQt::PVSourceWorkspace*)) {
				return false;
			}
			PVGuiQt::PVSourceWorkspace* workspace =
			    *(reinterpret_cast<PVGuiQt::PVSourceWorkspace* const*>(byte_array.constData()));

			_projects_tab_widget->add_workspace(workspace);

			return true;
		}
	}

	return QMainWindow::event(event);
}

// These methods are intentionally put in PVMainWindow's implementation
// as this might change in the near future and save lots of compilation time.
Squey::PVRoot& App::PVMainWindow::get_root()
{
	return _root;
}

Squey::PVRoot const& App::PVMainWindow::get_root() const
{
	return _root;
}

///////////////////////////////////////////////////////////////////////////

/******************************************************************************
 *
 * App::PVMainWindow::closeEvent
 *
 *****************************************************************************/
void App::PVMainWindow::closeEvent(QCloseEvent* event)
{
	if (_import_controller->busy()) {
		_projects_tab_widget->import_page().add_message(
		    PVGuiQt::PVImportPage::Severity::information,
		    tr("Squey closes once the import is over, or discarded."));
		event->ignore();
		return;
	}
	if (maybe_save_solution()) {
		event->accept();
	} else {
		event->ignore();
	}
}

/******************************************************************************
 *
 * App::PVMainWindow::commit_selection_to_new_layer
 *
 *****************************************************************************/
void App::PVMainWindow::commit_selection_to_new_layer(Squey::PVView* view)
{
	bool& should_hide_layers = view->edit_layer_stack().should_hide_layers();
	QString name = PVWidgets::PVNewLayerDialog::get_new_layer_name_from_dialog(
	    view->get_layer_stack().get_new_layer_name(), should_hide_layers, this);

	if (name.isEmpty()) {
		return;
	}

	Squey::PVAnalysisHistory::Scope step(*view, tr("New layer \"%1\" from the selection").arg(name), "layer-from-selection");
	view->commit_selection_to_new_layer(name, should_hide_layers);
}

/******************************************************************************
 *
 * App::PVMainWindow::move_selection_to_new_layer
 *
 *****************************************************************************/
void App::PVMainWindow::move_selection_to_new_layer(Squey::PVView* squey_view)
{
	Squey::PVLayer& current_layer = squey_view->edit_current_layer();

	bool& should_hide_layers = squey_view->edit_layer_stack().should_hide_layers();
	QString name = PVWidgets::PVNewLayerDialog::get_new_layer_name_from_dialog(
	    squey_view->get_layer_stack().get_new_layer_name(), should_hide_layers, this);

	if (!name.isEmpty()) {
		Squey::PVAnalysisHistory::Scope step(
		    *squey_view, tr("Selection moved to layer \"%1\"").arg(name), "move-layer-from-selection");

		if (should_hide_layers) {
			squey_view->hide_layers();
		}

		squey_view->add_new_layer(name);
		Squey::PVLayer& new_layer = squey_view->edit_current_layer();

		/* We set it's selection to the final selection */
		squey_view->commit_selection_to_layer(new_layer);

		// We remove that selection from the current layer if it is not locked.
		if (not current_layer.is_locked()) {
			current_layer.get_selection().and_not(new_layer.get_selection());
		}

		/* We need to reprocess the layer stack */
		squey_view->update_current_layer_min_max();
		squey_view->compute_selectable_count(new_layer);

		// do not forget to update the current layer
		squey_view->compute_selectable_count(current_layer);

		squey_view->process_layer_stack();
	}
}

/******************************************************************************
 *
 * App::PVMainWindow::close_solution_Slot
 *
 *****************************************************************************/
void App::PVMainWindow::close_solution_Slot()
{
	reset_root();
	set_window_title_with_filename();
}

/******************************************************************************
 *
 * App::PVMainWindow::import_type
 *
 *****************************************************************************/
void App::PVMainWindow::import_type(PVRush::PVInputType_p in_t)
{
	// For what it asks to edit a format with.
	in_t->connect_parent(this);
	if (PVWidgets::PVImportWidget* content = in_t->create_import_widget(nullptr)) {
		_import_controller->edit(in_t, content);
	}
}

/******************************************************************************
 *
 * App::PVMainWindow::import_type_default_Slot
 *
 *****************************************************************************/
void App::PVMainWindow::import_type_default_Slot()
{
	import_type(LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("file"));
}

/******************************************************************************
 *
 * App::PVMainWindow::import_type_Slot
 *
 *****************************************************************************/
void App::PVMainWindow::import_type_Slot()
{
	auto* action_src = (QAction*)sender();
	QString const& itype = action_src->data().toString();
	import_type_Slot(itype);
}

void App::PVMainWindow::import_type_Slot(const QString& itype)
{
	PVRush::PVInputType_p in_t = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name(itype);
	import_type(in_t);
}

/******************************************************************************
 *
 * App::PVMainWindow::sample_dataset_path
 *
 *****************************************************************************/
QString App::PVMainWindow::sample_dataset_path() const
{
	// Canonical, because this path is written down and shown back to the user in
	// the recent sources: shared_dir() may well have reached it by going up and
	// back down again.
	return QFileInfo(PVCore::PVConfig::shared_dir() + "/" + SQUEY_SAMPLE_DATASET_FILENAME)
	    .canonicalFilePath();
}

/******************************************************************************
 *
 * App::PVMainWindow::load_sample_dataset_Slot
 *
 *****************************************************************************/
void App::PVMainWindow::load_sample_dataset_Slot()
{
	const QString sample = sample_dataset_path();
	if (sample.isEmpty()) {
		return; // the action is disabled in that case
	}

	// Being a Parquet file, its format comes from the file itself: this opens
	// straight into the views, with nothing to describe on the way.
	load_files({sample});
}

/******************************************************************************
 *
 * App::PVMainWindow::register_sample_dataset
 *
 *****************************************************************************/
void App::PVMainWindow::register_sample_dataset()
{
	// Offered once. A user who cleared it away from the recent sources meant to,
	// and a machine that has already been used has recent sources of its own to
	// show.
	static const QString config_key = "squey/sample_dataset_offered";
	if (PVCore::PVConfig::get().config().value(config_key, false).toBool()) {
		return;
	}

	const QString sample = sample_dataset_path();
	if (sample.isEmpty()) {
		return; // a build that was run without being installed
	}

	try {
		PVRush::PVInputType_p in_t =
		    LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("parquet");

		// Being a Parquet file, its format is derived from the file's own schema.
		// That is what lets the entry be complete -- and openable in one click --
		// without anything having been imported first.
		PVRush::PVInputType::list_inputs inputs;
		PVRush::PVFormat format;
		const QString params = QJsonDocument(QJsonObject{{"paths", QJsonArray{sample}}})
		                           .toJson(QJsonDocument::Compact);
		PVCore::PVTaskProgress progress;
		if (not in_t->create_source_description_params(params, inputs, format, progress)) {
			return;
		}

		PVCore::PVRecentItemsManager::get().add_source(
		    PVRush::PVSourceCreatorFactory::get_by_input_type(in_t), inputs, format);
	} catch (const std::runtime_error& e) {
		PVLOG_WARN("Could not offer the sample dataset: %s\n", e.what());
		return;
	}

	PVCore::PVConfig::get().config().setValue(config_key, true);
	PVCore::PVConfig::get().config().sync();
}

/******************************************************************************
 *
 * App::PVMainWindow::load_files
 *
 *****************************************************************************/
void App::PVMainWindow::load_files(QStringList const& files, QString format_str /* = "" */)
{
	// Files handed over by another instance, while an import runs: their importer may ask about
	// them, which waits for the end of the import.
	if (_import_controller->busy()) {
		connect(
		    _import_controller.get(), &PVImportController::idle, this,
		    [this, files, format_str]() { load_files(files, format_str); },
		    Qt::SingleShotConnection);
		return;
	}

	// Nobody has checked these paths before they got here: they come from a
	// command line, or from a desktop that handed over a file this process
	// cannot reach -- under flatpak, anything in /tmp is outside the sandbox.
	// Report those, and go on with the rest rather than refusing the batch.
	QStringList existing_files;
	QStringList missing_files;
	for (QString const& file : files) {
		const QFileInfo fi(file);
		if (fi.exists()) {
			// Absolute: a relative path handed over by another instance was
			// written against *its* working directory, not ours.
			existing_files << fi.absoluteFilePath();
		} else {
			missing_files << file;
		}
	}
	if (not missing_files.isEmpty()) {
		QString explanation = tr("The following file(s) do not exist, and are ignored:\n\n%1")
		                          .arg(missing_files.join("\n"));
		// A flatpak has a /tmp, and a few other directories, of its own: a file
		// plainly there for the file manager that opened it is not there for
		// us. Worth saying, the path being reported as missing while the user
		// is looking at it.
		if (QFile::exists("/.flatpak-info")) {
			explanation +=
			    tr("\n\nThis application runs in a sandbox: files kept outside of the home "
			       "directory, such as in /tmp, are not visible to it.");
		}
		QMessageBox::critical(this, tr("File not found"), explanation, QMessageBox::Ok);
	}
	if (existing_files.isEmpty()) {
		return;
	}

	// An investigation opens in a window of its own -- one per file, and the
	// window of one that is already open is simply brought forward. Data files
	// go through an importer instead, so the two cannot be mixed in one batch.
	QStringList investigations;
	QStringList sources;
	for (QString const& file : existing_files) {
		if (QFileInfo(file).suffix().compare("pvi", Qt::CaseInsensitive) == 0) {
			investigations << file;
		} else {
			sources << file;
		}
	}
	if (not investigations.isEmpty() and not sources.isEmpty()) {
		QMessageBox::critical(
			this, tr("Mismatching file types"),
			tr("An investigation cannot be opened together with data files."),
			QMessageBox::Ok);
		return;
	}
	if (not investigations.isEmpty()) {
		for (QString const& investigation : investigations) {
			load_solution_and_create_mw(investigation);
		}
		return;
	}

	// One importer for the whole batch, resolved before anything is asked of
	// the user: a mismatch is worth reporting straight away rather than after a
	// question about files that will not be loaded. Which extensions go
	// together is the plugins' business -- .pcap and .pcapng are the same
	// importer, .csv and .csv.gz too.
	PVRush::PVInputType_p in_file;
	try {
		for (QString const& file : sources) {
			PVRush::PVInputType_p in_t =
			    LIB_CLASS(PVRush::PVInputType)::get().get_class_by_extension(file);
			if (not in_file) {
				in_file = in_t;
			} else if (in_t.get() != in_file.get()) {
				QMessageBox::critical(
					this, tr("Mismatching file types"),
					tr("Only files handled by a single importer can be loaded together."),
					QMessageBox::Ok);
				return;
			}
		}
	} catch (std::runtime_error const& e) {
		QMessageBox::critical(this, tr("Error when loading file(s)"),
		                      QString::fromStdString(e.what()), QMessageBox::Ok);
		return;
	}

	// What the import page asks for this type tells how many sources the files make.
	in_file->connect_parent(this);
	if (PVWidgets::PVImportWidget* content = in_file->create_import_widget(nullptr)) {
		if (_import_controller->edit(in_file, content)) {
			content->set_paths(sources);
			if (not format_str.isEmpty()) {
				content->set_format(format_str);
			}
		}
	}
}

void App::PVMainWindow::display_inv_elts()
{
	if (current_view()) {
		if (current_view()->get_parent<Squey::PVSource>().get_invalid_evts().size() > 0) {
			PVGuiQt::PVWorkspaceBase* workspace = _projects_tab_widget->current_workspace();
			if (auto* source_workspace =
			        dynamic_cast<PVGuiQt::PVSourceWorkspace*>(workspace)) {
				source_workspace->get_source_invalid_evts_dlg()->show();
			}
		} else {
			QMessageBox::information(this, tr("Invalid events"),
			                         tr("No invalid events have been saved or created during the "
			                            "extraction of this source."));
		}
	}
}

/******************************************************************************
 * App::PVMainWindow::save_screenshot
 *****************************************************************************/

void App::PVMainWindow::save_screenshot(const QPixmap& pixmap,
                                                const QString& title,
                                                const QString& name)
{
	QString filename = "screenshot_" + name;

	static const QString default_prefix("_0001.png");

	if (_screenshot_root_dir.isEmpty()) {
		_screenshot_root_dir = QDir::currentPath();
	}

	/**
	 * we get the last filename matching the "prefix" and we
	 * try to extract its counter.
	 */
	QDir dir(_screenshot_root_dir, filename + "_*.png");
	QStringList fnl =
	    dir.entryList(QDir::Files | QDir::NoDotAndDotDot, QDir::Name | QDir::Reversed);

	if (fnl.isEmpty() == false) {
		QRegExp re(filename + "_(\\d+).*");
		int pos = re.indexIn(fnl[0], 0);
		if (pos != -1) {
			int count = re.cap(1).toInt() + 1;
			filename += QString("_%1.png").arg(count, 4, 10, QChar('0'));
		} else {
			filename.append(default_prefix);
		}
	} else {
		filename.append(default_prefix);
	}

	QString img_name = PVWidgets::PVFileDialog::getSaveFileName(
	    this, title, QDir(_screenshot_root_dir).filePath(filename), QString("PNG Image (*.png)"));

	if (img_name.isEmpty()) {
		return;
	}

	// The portal grants the file of the name typed: another one would land beside it, unseen.
	if (img_name.endsWith(".png") == false and
	    not PVCore::PVFlatpakHelper::is_document_portal_path(img_name)) {
		img_name += ".png";
	}

	_screenshot_root_dir = PVCore::PVFlatpakHelper::host_directory_of(img_name);

	if (pixmap.save(img_name, "PNG") == false) {
		QMessageBox::critical(this, "Error saving the screenshot",
		                      "Check for permissions in '" + _screenshot_root_dir +
		                          "' or for free disk space",
		                      QMessageBox::Ok);
	}
}

static size_t invalid_columns_count(const Squey::PVSource* src)
{
	const PVRush::PVNraw& nraw = src->get_rushnraw();

	size_t invalid_columns_count = 0;
	for (PVCol col(0); col < nraw.column_count(); col++) {
		invalid_columns_count +=
		    bool(nraw.column(col).has_invalid() & pvcop::db::INVALID_TYPE::INVALID);
	}

	return invalid_columns_count;
}

static QString bad_conversions_as_string(const Squey::PVSource* src)
{
	QStringList l;

	auto const& ax = src->get_format().get_axes();
	const PVRush::PVNraw& nraw = src->get_rushnraw();

	// We must cap the number of invalid values and their length because Qt could crash otherwise
	size_t max_total_size = 10000;
	size_t max_values = 1000;

	size_t total_size = 0;
	bool end = false;

	for (size_t row = 0; row < nraw.row_count() and not end; row++) {
		for (PVCol col(0); col < nraw.column_count() and not end; col++) {

			const pvcop::db::array& column = nraw.column(col);
			if (not(column.has_invalid() & pvcop::db::INVALID_TYPE::INVALID)) {
				continue;
			}

			if (not column.is_valid(row)) {
				const std::string invalid_value = column.at(row);

				if (invalid_value == "") {
					continue;
				}

				QString str("row #" + QString::number(row + 1) + " :");
				const QString& axis_name = ax[col].get_name();
				const QString& axis_type = ax[col].get_type();

				str += " " + axis_name + " (" + axis_type + ") : \"" +
				       QString::fromStdString(invalid_value) + "\"";

				l << str;

				total_size += str.size();

				if (max_values-- == 0 or total_size > max_total_size) {
					l << "There are more errors but only the first are shown. You should edit your "
					     "format and specify the proper types.";
					end = true;
				}
			}
		}
	}

	return l.join("\n");
}

void App::PVMainWindow::source_loaded(Squey::PVSource& src, bool update_recent_items)
{
	// Create workspace for this source.
	_projects_tab_widget->add_source(&src);

	// Show invalide elements.
	if (src.get_invalid_evts().size() > 0) {
		display_inv_elts();
	}

	/**
	 * A source can be added to the recent items as soon as its format can be
	 * retrieved on reload: either it is saved on disk, or it is generated from
	 * the inputs (e.g. Parquet) and can be regenerated when reloading.
	 */
	const QString format_path = src.get_original_format().get_full_path();
	const bool format_reloadable =
	    not format_path.isEmpty() or
	    src.get_source_creator()->supported_type_lib()->has_generated_format();

	if (update_recent_items and format_reloadable and
	    src.get_source_creator()->name() != "pcap") {
		// Add format as recent format (only relevant when saved on disk)
		if (not format_path.isEmpty()) {
			PVCore::PVRecentItemsManager::get().add<PVCore::Category::USED_FORMATS>(format_path);
		}

		// Add source as recent source
		PVCore::PVRecentItemsManager::get().add_source(src.get_source_creator(), src.get_inputs(),
		                                               src.get_original_format());
	}

	// Execute Python script if any
	bool is_path, disabled;
	QString python_script = src.get_original_format().get_python_script(is_path, disabled);
	if (not disabled and not python_script.isEmpty()) {
		if (is_path and not QFileInfo(python_script).exists()) {
			QMessageBox::warning(this, tr("Unable to execute Python script"),
				python_script + tr(" is missing"), QMessageBox::Ok);
		}
		else {
#ifdef PYTHON_SUPPORT
			QString exception_message;
			Squey::PVPythonInterpreter& python_interpreter = Squey::PVPythonInterpreter::get(_root);
			PVCore::PVProgressBox::CancelState cancel_state = PVGuiQt::PVProgressBoxPython::progress([&](PVCore::PVProgressBox& pbox) {
				pbox.set_enable_cancel(true);
				try {
					python_interpreter.execute_script(python_script.toStdString(), is_path);
				}
				catch (const pybind11::error_already_set &eas) {
					throw eas; // rethrow exception to handle progress box dismiss
				}
			}, current_view(), QString("Executing python script"), exception_message, this);

			if (cancel_state == PVCore::PVProgressBox::CancelState::CONTINUE and not exception_message.isEmpty()) {
				QMessageBox::critical(this, "Error while executing Python script", exception_message , QMessageBox::Ok);
			}
#endif
		}
	}
}

/******************************************************************************
 *
 * App::PVMainWindow::remove_source
 *
 *****************************************************************************/
void App::PVMainWindow::remove_source(Squey::PVSource* src_p)
{
	Squey::PVScene& scene_p = src_p->get_parent();

	scene_p.remove_child(*src_p);
	if (scene_p.size() == 0) {
		PVGuiQt::PVSceneWorkspacesTabWidget* tab =
		    _projects_tab_widget->get_workspace_tab_widget_from_scene(&scene_p);
		_projects_tab_widget->remove_project(tab);
		tab->deleteLater();
	}
}

/******************************************************************************
 *
 * App::PVMainWindow::set_color()
 *
 *****************************************************************************/
void App::PVMainWindow::set_color(Squey::PVView* squey_view)
{
	PVLOG_DEBUG("App::PVMainWindow::%s\n", __FUNCTION__);

	/* We let the user select a color */
	PVWidgets::PVColorDialog dial;
	if (dial.exec() != QDialog::Accepted) {
		return;
	}

	PVCore::PVHSVColor color = dial.color();

	Squey::PVAnalysisHistory::Scope step(*squey_view, tr("Colour the selected events"), "palette");
	squey_view->set_color_on_active_layer(color);
}

void App::PVMainWindow::reset_root()
{
	get_root().clear();
}

void App::PVMainWindow::close_solution()
{
	if (!maybe_save_solution()) {
		return;
	}
	close_solution_Slot();
}

// Mainly from Qt's SDI example
App::PVMainWindow* App::PVMainWindow::find_main_window(const QString& path)
{
	QString canonicalFilePath = QFileInfo(path).canonicalFilePath();

	for (QWidget* widget : qApp->topLevelWidgets()) {
		auto* mw = qobject_cast<PVMainWindow*>(widget);
		if (mw && QFileInfo(mw->get_solution_path()).canonicalFilePath() == canonicalFilePath)
			return mw;
	}
	return nullptr;
}
