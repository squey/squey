/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <QApplication>
#include <QDesktopWidget>
#include <QDialog>
#include <QFile>
#include <QLabel>
#include <QMessageBox>
#include <QStatusBar>
#include <QVBoxLayout>

#include <PVMainWindow.h>
#include <PVStringListChooserWidget.h>

#include <pvguiqt/PVWorkspace.h>
#include <inendi/widgets/PVNewLayerDialog.h>

#include <pvkernel/core/PVRecentItemsManager.h>
#include <pvkernel/core/PVConfig.h>
#include <pvkernel/core/inendi_bench.h>

#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVMeanValue.h>
#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/core/PVWSLHelper.h>

#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVNrawException.h>
#include <pvkernel/rush/PVUnicodeSourceError.h>
#include <pvkernel/rush/PVConverter.h>

#include <pvkernel/widgets/PVColorDialog.h>
#include <pvkernel/widgets/PVFileDialog.h>

#include <inendi/PVSelection.h>
#include <inendi/PVStateMachine.h>
#include <inendi/PVSource.h>
#include <inendi/PVAxis.h>

#include <pvparallelview/PVParallelView.h>
#include <pvguiqt/PVExportSelectionDlg.h>

#include <PVFormatBuilderWidget.h>

#include <tbb/tick_count.h>

#include <boost/thread.hpp>

#include <sys/utsname.h> // uname

/******************************************************************************
 *
 * PVInspector::PVMainWindow::PVMainWindow
 *
 *****************************************************************************/
PVInspector::PVMainWindow::PVMainWindow(QWidget* parent)
    : QMainWindow(parent)
    , _load_solution_dlg(this,
                         tr("Load an investigation..."),
                         QString(),
                         INENDI_ROOT_ARCHIVE_FILTER ";;" ALL_FILES_FILTER)
    , _root()
{
	setAcceptDrops(true);

	reset_root();

	// OBJECTNAME STUFF
	setObjectName("PVMainWindow");

	// FONT stuff
	QFontDatabase pv_font_database;

	// FIXME: check fonts licenses to be sure we can distribute them with Inspector
	pv_font_database.addApplicationFont(QString(":/Convergence-Regular.ttf"));
	pv_font_database.addApplicationFont(QString(":/Jura-DemiBold.ttf"));
	pv_font_database.addApplicationFont(QString(":/OSP-DIN.ttf"));
	pv_font_database.addApplicationFont(QString(":/PT_Sans-Narrow-Web-Bold.ttf"));
	pv_font_database.addApplicationFont(QString(":/PT_Sans-Narrow-Web-Regular.ttf"));

	// import_source = nullptr;
	report_started = false;
	report_image_index = 0;
	report_filename = nullptr;

	// We activate all available Windows
	_projects_tab_widget = new PVGuiQt::PVProjectsTabWidget(&get_root());
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

	// We display the PV Icon together with a button to import files
	pv_centralMainWidget = new QWidget();
	pv_centralMainWidget->setObjectName("pv_centralMainWidget_of_PVMainWindow");

	pv_mainLayout = new QVBoxLayout();
	pv_mainLayout->setContentsMargins(0, 0, 0, 0);

	pv_mainLayout->addWidget(_projects_tab_widget);

	/**
	 * Show warning message when no GPU accelerated device has been found
	 * Except under WSL where GPU is not supported yet
	 * (https://wpdev.uservoice.com/forums/266908-command-prompt-console-windows-subsystem-for-l/suggestions/16108045-opencl-cuda-gpu-support)
	 */
	if (not PVParallelView::common::is_gpu_accelerated() and
	    not PVCore::PVWSLHelper::is_microsoft_wsl()) {
		/* the warning icon
		 */
		QIcon warning_icon = QApplication::style()->standardIcon(QStyle::SP_MessageBoxWarning);
		QLabel* warning_label_icon = new QLabel;
		warning_label_icon->setPixmap(warning_icon.pixmap(QSize(16, 16)));
		statusBar()->addPermanentWidget(warning_label_icon, 0);

		/* and the message
		 */
		QLabel* warning_msg = new QLabel("<font color=\"orange\"><b>You are running in degraded "
		                                 "mode without GPU acceleration. </b></font>");
		statusBar()->addPermanentWidget(warning_msg, 0);
	}

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

	// Center the main window
	QRect r = geometry();
	r.moveCenter(QApplication::desktop()->screenGeometry(this).center());
	setGeometry(r);

	// Set stylesheet
	QFile css_file(":/gui.css");
	css_file.open(QFile::ReadOnly);
	QTextStream css_stream(&css_file);
	QString css_string(css_stream.readAll());
	css_file.close();
	setStyleSheet(css_string);

	showMaximized();
}

bool PVInspector::PVMainWindow::event(QEvent* event)
{
	QString mime_type = "application/x-inendi_workspace";

	if (event->type() == QEvent::DragEnter) {
		QDragEnterEvent* dragEnterEvent = static_cast<QDragEnterEvent*>(event);
		if (dragEnterEvent->mimeData()->hasFormat(mime_type)) {
			dragEnterEvent->accept(); // dragEnterEvent->acceptProposedAction();
			return true;
		}
	} else if (event->type() == QEvent::Drop) {
		QDropEvent* dropEvent = static_cast<QDropEvent*>(event);
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
Inendi::PVRoot& PVInspector::PVMainWindow::get_root()
{
	return _root;
}

Inendi::PVRoot const& PVInspector::PVMainWindow::get_root() const
{
	return _root;
}

///////////////////////////////////////////////////////////////////////////

/******************************************************************************
 *
 * PVInspector::PVMainWindow::closeEvent
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::closeEvent(QCloseEvent* event)
{
	if (maybe_save_solution()) {
		event->accept();
	} else {
		event->ignore();
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::commit_selection_to_new_layer
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::commit_selection_to_new_layer(Inendi::PVView* inendi_view)
{
	bool& should_hide_layers = inendi_view->get_layer_stack().should_hide_layers();
	QString name = PVWidgets::PVNewLayerDialog::get_new_layer_name_from_dialog(
	    inendi_view->get_layer_stack().get_new_layer_name(), should_hide_layers, this);

	if (name.isEmpty()) {
		return;
	}

	if (should_hide_layers) {
		inendi_view->hide_layers();
	}

	inendi_view->add_new_layer(name);
	Inendi::PVLayer& layer = inendi_view->get_current_layer();

	// We need to configure the layer
	inendi_view->commit_selection_to_layer(layer);
	inendi_view->update_current_layer_min_max();
	inendi_view->compute_selectable_count(layer);
	inendi_view->process_layer_stack();
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::move_selection_to_new_layer
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::move_selection_to_new_layer(Inendi::PVView* inendi_view)
{
	Inendi::PVLayer& current_layer = inendi_view->get_current_layer();

	bool& should_hide_layers = inendi_view->get_layer_stack().should_hide_layers();
	QString name = PVWidgets::PVNewLayerDialog::get_new_layer_name_from_dialog(
	    inendi_view->get_layer_stack().get_new_layer_name(), should_hide_layers, this);

	if (!name.isEmpty()) {

		if (should_hide_layers) {
			inendi_view->hide_layers();
		}

		inendi_view->add_new_layer(name);
		Inendi::PVLayer& new_layer = inendi_view->get_current_layer();

		/* We set it's selection to the final selection */
		inendi_view->commit_selection_to_layer(new_layer);

		// We remove that selection from the current layer if it is not locked.
		if (not current_layer.is_locked()) {
			current_layer.get_selection().and_not(new_layer.get_selection());
		}

		/* We need to reprocess the layer stack */
		inendi_view->update_current_layer_min_max();
		inendi_view->compute_selectable_count(new_layer);

		// do not forget to update the current layer
		inendi_view->compute_selectable_count(current_layer);

		inendi_view->process_layer_stack();
	}
}

// Check if we have already a menu with this name at this level
static QMenu* create_filters_menu_exists(QHash<QMenu*, int> actions_list, QString name, int level)
{
	QHashIterator<QMenu*, int> iter(actions_list);
	while (iter.hasNext()) {
		iter.next();
		QString menu_title = iter.key()->title();
		int menu_level = iter.value();

		if ((!menu_title.compare(name)) && (menu_level == level)) {
			return iter.key();
		}
	}

	return nullptr;
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::create_filters_menu_and_actions
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::create_filters_menu_and_actions()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);

	QMenu* menu = filter_Menu;
	QHash<QMenu*, int> actions_list; // key = action name; value = menu level;
	                                 // Foo/Bar/Camp makes Foo at level 0, Bar at
	                                 // level 1, etc.

	LIB_CLASS(Inendi::PVLayerFilter)& filters_layer = LIB_CLASS(Inendi::PVLayerFilter)::get();
	LIB_CLASS(Inendi::PVLayerFilter)::list_classes const& lf = filters_layer.get_list();
	LIB_CLASS(Inendi::PVLayerFilter)::list_classes::const_iterator it;

	for (it = lf.begin(); it != lf.end(); it++) {
		//(*it).get_args()["Menu_name"]
		QString filter_name = it->key();
		QString action_name = it->value()->menu_name();
		QString status_tip = it->value()->status_bar_description();

		QStringList actions_name = action_name.split(QString("/"));
		if (actions_name.count() > 1) {
			// // qDebug("actions_name[0]=%s\n", qPrintable(actions_name[0]));
			// // We add the various submenus
			for (int i = 0; i < actions_name.count(); i++) {
				bool is_last = i == actions_name.count() - 1 ? 1 : 0;

				// Step 1: we add the different menus into the hash
				QMenu* menu_exists = create_filters_menu_exists(actions_list, actions_name[i], i);
				if (!menu_exists) {
					QMenu* filter_element_menu = new QMenu(actions_name[i]);
					actions_list[filter_element_menu] = i;
				}

				// Step 2: we connect the menus with each other and connect the actions
				QMenu* menu_to_add = create_filters_menu_exists(actions_list, actions_name[i], i);
				if (!menu_to_add) {
					PVLOG_ERROR("The menu named '%s' at position level %d cannot be "
					            "added since it was not append previously!\n",
					            qPrintable(actions_name[i]), i);
				}
				if (i == 0) { // We are at root level
					menu->addMenu(menu_to_add);
				} else {
					if (is_last) {
						QMenu* previous_menu =
						    create_filters_menu_exists(actions_list, actions_name[i - 1], i - 1);

						QAction* action = new QAction(actions_name[i] + "...", previous_menu);
						action->setObjectName(filter_name);
						action->setStatusTip(status_tip);
						connect(action, &QAction::triggered, this, &PVMainWindow::filter_Slot);
						previous_menu->addAction(action);
					} else {
						// we add a menu to the previous menu
						QMenu* previous_menu =
						    create_filters_menu_exists(actions_list, actions_name[i - 1], i - 1);
						QMenu* current_menu =
						    create_filters_menu_exists(actions_list, actions_name[i], i);
						previous_menu->addMenu(current_menu);
					}
				}
			}
		} else { // Nothing to split, so there is only a direct action
			QAction* action = new QAction(action_name + "...", menu);
			action->setObjectName(filter_name);
			action->setStatusTip(status_tip);
			connect(action, &QAction::triggered, this, &PVMainWindow::filter_Slot);

			menu->addAction(action);
		}
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::close_solution_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::close_solution_Slot()
{
	reset_root();
	set_window_title_with_filename();
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::import_type
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::import_type(PVRush::PVInputType_p in_t)
{
	PVRush::PVSourceCreator_p src_creator = PVRush::PVSourceCreatorFactory::get_by_input_type(in_t);
	PVRush::hash_format_creator format_creator;

	PVRush::hash_formats formats;

	// Create the input widget
	QString choosenFormat;
	// PVInputType::list_inputs is a QList<PVInputDescription_p>
	PVRush::PVInputType::list_inputs inputs;

	PVCore::PVArgumentList args;

	if (!in_t->createWidget(formats, inputs, choosenFormat, args, this))
		return; // This means that the user pressed the "cancel" button

	// Add the new formats to the formats
	{
		for (auto it = formats.begin(); it != formats.end(); it++) {
			PVRush::hash_format_creator::mapped_type v(it.value(), src_creator);
			// Save this format/creator pair to the "format_creator" object
			format_creator[it.key()] = v;
		}
	}

	import_type(in_t, inputs, formats, format_creator, choosenFormat);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::import_type
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::import_type(PVRush::PVInputType_p in_t,
                                            PVRush::PVInputType::list_inputs const& inputs,
                                            PVRush::hash_formats& formats,
                                            PVRush::hash_format_creator& format_creator,
                                            QString const& choosenFormat)
{
	PVRush::PVSourceCreator_p sc = PVRush::PVSourceCreatorFactory::get_by_input_type(in_t);

	QHash<QString, PVRush::PVInputType::list_inputs> discovered;
	QHash<QString, std::pair<QString, QString>> formats_error; // Errors w/ some formats

	QHash<QString, PVRush::PVInputDescription_p> hash_input_name;

	bool file_type_found = false;

	try {
		if (choosenFormat.compare(INENDI_LOCAL_FORMAT_STR) == 0) {
			PVRush::hash_formats custom_formats;
			PVRush::list_creators pre_discovered_creators;

			for (auto input_it = inputs.begin(); input_it != inputs.end(); ++input_it) {
				QString in_str = (*input_it)->human_name();
				hash_input_name[in_str] = *input_it;

				pre_discovered_creators.push_back(sc);
				in_t->get_custom_formats(*input_it, custom_formats);

				for (auto hf_it = custom_formats.begin(); hf_it != custom_formats.end(); ++hf_it) {
					formats.insert(hf_it.key(), hf_it.value());

					PVRush::hash_format_creator::mapped_type v(hf_it.value(), sc);
					format_creator[hf_it.key()] = v;
				}
			}

			if (custom_formats.size() == 1) {
				file_type_found = true;
				discovered[custom_formats.keys()[0]] = inputs;
			}
		} else if (choosenFormat.compare(INENDI_BROWSE_FORMAT_STR) == 0) {
			file_type_found = false;
		} else if (choosenFormat == "custom") {
			file_type_found = true;
			discovered["custom"] = inputs;
		} else {
			QFileInfo fi(choosenFormat);
			QString format_name = choosenFormat;
			PVRush::PVFormat format(format_name, choosenFormat);
			formats[format_name] = format;
			PVRush::hash_format_creator::mapped_type v(format, sc);
			format_creator[format_name] = v;
			if (fi.isReadable()) {
				file_type_found = true;
				discovered[format_name] = inputs;
			}
		}
	} catch (const PVRush::PVInvalidFile& e) {
		QMessageBox::critical(this, tr("Fatal error while loading source..."), e.what());
		return;
	}

	treat_invalid_formats(formats_error);

	// First, try complete autodetection
	if (!file_type_found and choosenFormat.compare(INENDI_BROWSE_FORMAT_STR) != 0) {
		for (auto& input : inputs) {
			try {
				PVFormatBuilderWidget* editorWidget = new PVFormatBuilderWidget(this);
				editorWidget->show();
				PVRush::PVFormat guess_format =
				    editorWidget->load_log_and_guess_format(input, in_t);
				if (not guess_format.is_valid() or not editorWidget->close()) {
					PVLOG_ERROR("Could not autodetect format.");
					QDialog editor_dialog(this);
					editor_dialog.setModal(true);
					editor_dialog.resize(editorWidget->size());
					QLayout* l = new QHBoxLayout();
					l->addWidget(editorWidget);
					editor_dialog.setLayout(l);
					editor_dialog.exec();
					guess_format = editorWidget->get_format_from_dom();
				}
				auto format_name = editorWidget->get_current_format_name();
				if (format_name.isEmpty() or not QFile::exists(format_name)) {
					PVLOG_ERROR("Format not saved.");
					break;
				}
				guess_format.set_full_path(format_name);
				formats[format_name] = guess_format;
				PVRush::hash_format_creator::mapped_type v(guess_format, sc);
				format_creator[format_name] = v;
				discovered[format_name] << input;
				file_type_found = true;
			} catch (const PVRush::PVInvalidFile& e) {
				QMessageBox::critical(this, tr("Fatal error while loading source..."), e.what());
				break;
			}
		}
	}

	if (!file_type_found) {

		/* A QFileDialog is explicitly used over QFileDialog::getOpenFileName
		 * because this latter does not used QFileDialog's global environment
		 * like last used current directory.
		 */

		PVWidgets::PVFileDialog* fdialog = new PVWidgets::PVFileDialog(this);

		fdialog->setNameFilter("Formats (*.format)");
		fdialog->setWindowTitle("Load format from...");

		int ret = fdialog->exec();

		if (ret == QDialog::Accepted) {
			QString format_path = fdialog->selectedFiles().at(0);
			QFileInfo fi(format_path);
			QString format_name = fi.dir().path();

			PVRush::PVFormat format(format_name, format_path);
			formats[format_name] = format;
			PVRush::hash_format_creator::mapped_type v(format, sc);
			format_creator[format_name] = v;

			if (fi.isReadable()) {
				file_type_found = true;
				discovered[format_name] = inputs;
			}
		}

		delete fdialog;

		if (ret == QDialog::Rejected) {
			return;
		}
	}

	bool one_extraction_successful = false;
	QStringList invalid_formats;
	// Load a type of file per view

	/* can not use a C++11 foreach because QHash<...>::const_iterator is not
	 * an usual const iterator but a non-const iterator which behaves as a
	 * const one... I hate Qt!
	 */
	for (auto it = discovered.constBegin(); it != discovered.constEnd(); it++) {
		// Create scene and source

		const PVRush::PVInputType::list_inputs& inputs = it.value();

		size_t input_index = 0;
		for (PVRush::PVFormat const& format : formats) {
			PVRush::PVInputType::list_inputs in;

			if (formats.size() > 1) {
				in.append(inputs[input_index++]);
			} else {
				in = inputs;
			}

			PVRush::PVSourceDescription src_desc(in, sc, format);

			try {
				if (load_source_from_description_Slot(src_desc)) {
					one_extraction_successful = true;
				}
			} catch (Inendi::InvalidPlottingMapping const& e) {
				invalid_formats.append(it.key() + ": " + e.what());
			} catch (PVRush::PVInvalidFile const& e) {
				invalid_formats.append(it.key() + ": " + e.what());
			}
			catch (const std::runtime_error& e) {
				QMessageBox::critical(this, "Runtime error", e.what());
				return;
			}
		}

		if (not invalid_formats.isEmpty()) {
			QMessageBox error_message(
			    QMessageBox::Warning, "Invalid format",
			    "Some format can't be use as types, mapping and plotting are not compatible.",
			    QMessageBox::Ok, this);
			error_message.setDetailedText(invalid_formats.join("\n"));
			error_message.exec();
		}

		if (!one_extraction_successful) {
			return;
		}
	}

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
	QAction* action_src = (QAction*)sender();
	QString const& itype = action_src->data().toString();
	import_type_Slot(itype);
}

void PVInspector::PVMainWindow::import_type_Slot(const QString& itype)
{
	PVRush::PVInputType_p in_t = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name(itype);
	import_type(in_t);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::keyPressEvent()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::keyPressEvent(QKeyEvent* event)
{
	QMainWindow::keyPressEvent(event);
#ifdef INENDI_DEVELOPER_MODE
	switch (event->key()) {

	case Qt::Key_Dollar: {
		/*if (pv_WorkspacesTabWidget->currentIndex() == -1) {
		        break;
		}*/
		PVLOG_INFO("Reloading CSS\n");

		QFile css_file(INENDI_SOURCE_DIRECTORY "/gui-qt/src/resources/gui.css");
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
	PVRush::PVSourceCreator_p src_creator =
	    PVRush::PVSourceCreatorFactory::get_by_input_type(in_file);
	PVRush::hash_format_creator format_creator;

	PVRush::hash_formats formats;
	{
		for (auto itfc = format_creator.begin(); itfc != format_creator.end(); ++itfc) {
			formats[itfc.key()] = itfc.value().first;
		}
	}

	// Create PVFileDescription objects
	//

	PVRush::PVInputType::list_inputs files_in;
	{
		for (QString filename : files) {
			files_in.push_back(PVRush::PVInputDescription_p(
			    new PVRush::PVFileDescription(filename, files.size() > 1)));
		}
	}

	if (!format.isEmpty()) {
		PVRush::PVFormat new_format("custom:arg", format);
		formats["custom:arg"] = new_format;
		PVRush::hash_format_creator::mapped_type v(new_format, src_creator);
		// Save this format/creator pair to the "format_creator" object
		format_creator["custom:arg"] = v;
		format = "custom:arg";
	}

	import_type(in_file, files_in, formats, format_creator, format);
}

void PVInspector::PVMainWindow::display_inv_elts()
{
	if (current_view()) {
		if (current_view()->get_parent<Inendi::PVSource>().get_invalid_evts().size() > 0) {
			PVGuiQt::PVWorkspaceBase* workspace = _projects_tab_widget->current_workspace();
			if (PVGuiQt::PVSourceWorkspace* source_workspace =
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
 * PVInspector::PVMainWindow::save_screenshot
 *****************************************************************************/

void PVInspector::PVMainWindow::save_screenshot(const QPixmap& pixmap,
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

	QString img_name = PVWidgets::PVFileDialog::getSaveFileName(this, title, filename,
	                                                            QString("PNG Image (*.png)"));

	if (img_name.isEmpty()) {
		return;
	}

	if (img_name.endsWith(".png") == false) {
		img_name += ".png";
	}

	_screenshot_root_dir = QFileInfo(img_name).dir().path();

	if (pixmap.save(img_name) == false) {
		QMessageBox::critical(this, "Error saving the screenshot",
		                      "Check for permissions in '" + _screenshot_root_dir +
		                          "' or for free disk space",
		                      QMessageBox::Ok);
	}
}

static size_t invalid_columns_count(const Inendi::PVSource* src)
{
	const PVRush::PVNraw& nraw = src->get_rushnraw();

	size_t invalid_columns_count = 0;
	for (PVCol col(0); col < nraw.column_count(); col++) {
		invalid_columns_count +=
		    bool(nraw.column(col).has_invalid() & pvcop::db::INVALID_TYPE::INVALID);
	}

	return invalid_columns_count;
}

static QString bad_conversions_as_string(const Inendi::PVSource* src)
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

/******************************************************************************
 *
 * PVInspector::PVMainWindow::load_source
 *
 *****************************************************************************/
bool PVInspector::PVMainWindow::load_source(Inendi::PVSource* src,
                                            bool update_recent_items /*= true*/)
{
	// Load a created source
	// Extract the source
	BENCH_START(lff);

	auto ret = PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& pbox) {
		    pbox.set_cancel2_btn_text("Stop and process");
		    pbox.set_cancel_btn_text("Discard");
		    pbox.set_confirmation(true);
		    constexpr size_t mega = 1024 * 1024;
		    // set min and max to 0 to have an activity effect
		    pbox.set_maximum(src->max_size() / mega);

		    // PVCore::PVProgressBox::progress();
		    PVRush::PVControllerJob_p job_import;
		    try {
			    job_import = src->extract(0);
		    } catch (PVRush::PVInputException const& e) {
			    // If input file can't be opened
			    pbox.critical("Cannot create sources", QString("Error with input: ") + e.what());
			    pbox.set_canceled();
			    return;
		    } catch (PVRush::PVNrawException const& e) {
			    // If we can't create the NRaw folder for example
			    pbox.critical("Cannot create sources", QString("Error with nraw: ") + e.what());
			    pbox.set_canceled();
			    return;
		    }

		    try {
			    // launch a thread in order to update the status of the progress bar
			    while (job_import->running()) {
				    pbox.set_extended_status(
				        QString("Number of extracted events: %L1\nNumber of rejected events: %L2")
				            .arg(job_import->status())
				            .arg(job_import->rejected_elements()));
				    pbox.set_value(job_import->get_value() / mega);
				    boost::this_thread::interruption_point();
				    boost::this_thread::sleep(boost::posix_time::milliseconds(50));
			    }
		    } catch (boost::thread_interrupted) {
			    job_import->cancel();
		    }
		    if (pbox.get_cancel_state() == PVCore::PVProgressBox::CancelState::CANCEL) {
			    return;
		    }

		    try {
			    src->wait_extract_end(job_import);
		    } catch (PVRush::PVInputException const& e) {
			    pbox.critical("Cannot create sources", QString("Error with input: ") + e.what());
			    pbox.set_canceled();
			    return;
		    } catch (PVRush::UnicodeSourceError const&) {
			    pbox.critical("Cannot create sources",
			                  "File encoding does permit Inspector to perform extraction.");
			    pbox.set_canceled();
			    return;
		    } catch (PVRush::PVConverterCreationError const& e) {
			    pbox.critical("Unsupported charset", e.what());
			    pbox.set_canceled();
			    return;
		    }
	    },
	    QString("Extracting %1...").arg(src->get_format_name()), this);

	if (ret == PVCore::PVProgressBox::CancelState::CANCEL) {
		// If job is canceled, stop here
		return false;
	}

	if (src->get_rushnraw().row_count() == 0) {
		QString msg = QString("<p>The files <strong>%1</strong> using format "
		                      "<strong>%2</strong> cannot be opened. ")
		                  .arg(QString::fromStdString(src->get_name()))
		                  .arg(src->get_format_name());
		PVRow nelts = src->get_invalid_evts().size();
		if (nelts > 0) {
			msg += QString("Indeed, <strong>%1 elements</strong> have been extracted "
			               "but were <strong>all invalid</strong>.</p>")
			           .arg(nelts);
			msg += QString("<p>This is because one or more splitters and/or "
			               "filters defined in format <strong>%1</strong> reported "
			               "invalid events during the extraction.<br />")
			           .arg(src->get_format_name());
			msg += QString("You may have invalid regular expressions set in this "
			               "format, or simply all the events have been invalidated "
			               "by one or more filters thus no events matches your "
			               "criterias.</p>");
			msg += QString("<p>You might try to <strong>fix your format</strong> or "
			               "try to load <strong>another set of data</strong>.</p>");
		} else {
			msg += QString("Indeed, the sources <strong>were empty</strong> (empty "
			               "files, bad database query, etc...) because no elements "
			               "have been extracted.</p><p>You should try to load "
			               "another set of data.</p>");
		}
		QMessageBox::critical(this, "Cannot load sources", msg);
		return false;
	} else {
		size_t inv_col_count = invalid_columns_count(src);
		const QString& details = bad_conversions_as_string(src);
		if (inv_col_count > 0 and not details.isEmpty()) {
			// We can continue with it but user have to know that some values are
			// incorrect.
			QMessageBox warning_message(
			    QMessageBox::Warning, "Failed conversion(s)",
			    "\n" + QString::number(inv_col_count) + "/" +
			        QString::number(src->get_nraw_column_count()) +
			        " column(s) have some values that failed to be properly "
			        "converted from text to binary during import...",
			    QMessageBox::Ok, this);
			warning_message.setDetailedText(details);
			warning_message.exec();
		}
	}

	BENCH_STOP(lff);
#ifdef INENDI_DEVELOPER_MODE
	PVLOG_INFO("nraw created from data in %g sec\n", BENCH_END_TIME(lff));
#endif

	if (PVCore::PVProgressBox::progress(
	        [&](PVCore::PVProgressBox& pbox) {
		        pbox.set_maximum(3);

		        pbox.set_value(0);
		        pbox.set_extended_status("Computing mapping...");
		        auto& mapped = src->emplace_add_child();

		        pbox.set_value(1);
		        pbox.set_extended_status("Computing plotting...");
		        auto& plotted = mapped.emplace_add_child();

		        pbox.set_value(2);
		        pbox.set_extended_status("Creating views...");
		        plotted.emplace_add_child();

		        pbox.set_value(3);
	        },
	        tr("Processing..."), (QWidget*)this) != PVCore::PVProgressBox::CancelState::CONTINUE) {
		return false;
	}

	source_loaded(*src, update_recent_items);

	return true;
}

void PVInspector::PVMainWindow::source_loaded(Inendi::PVSource& src, bool update_recent_items)
{
	// Create workspace for this source.
	_projects_tab_widget->add_source(&src);

	// Show invalide elements.
	if (src.get_invalid_evts().size() > 0) {
		display_inv_elts();
	}

	/**
	 * For the moment we can't add sources that have an auto generated format
	 * not saved to disk.
	 */
	if (update_recent_items and not src.get_original_format().get_full_path().isEmpty() and
	    src.get_source_creator()->name() != "pcap") {
		// Add format as recent format
		PVCore::PVRecentItemsManager::get().add<PVCore::Category::USED_FORMATS>(
		    src.get_original_format().get_full_path());

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
			auto& python_interpreter = src.get_parent<Inendi::PVRoot>().python_interpreter();
			PVCore::PVProgressBox::progress([&](PVCore::PVProgressBox& pbox) {
				pbox.set_enable_cancel(false);
				try {
					python_interpreter.execute_script(python_script.toStdString(), is_path);
				}
				catch (pybind11::error_already_set &eas) {
					Q_EMIT pbox.warning_sig("Error while executing Python script", eas.what());
				}
			}, QString("Executing python script"), this);
		}
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::remove_source
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::remove_source(Inendi::PVSource* src_p)
{
	Inendi::PVScene& scene_p = src_p->get_parent();

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
 * PVInspector::PVMainWindow::set_color()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::set_color(Inendi::PVView* inendi_view)
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);

	/* We let the user select a color */
	PVWidgets::PVColorDialog dial;
	if (dial.exec() != QDialog::Accepted) {
		return;
	}

	PVCore::PVHSVColor color = dial.color();

	inendi_view->set_color_on_active_layer(color);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::treat_invalid_formats
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::treat_invalid_formats(
    QHash<QString, std::pair<QString, QString>> const& errors)
{
	if (errors.size() == 0) {
		return;
	}

	QSettings& pvconfig = PVCore::PVConfig::get().config();

	if (!pvconfig.value(PVCONFIG_FORMATS_SHOW_INVALID, PVCONFIG_FORMATS_SHOW_INVALID_DEFAULT)
	         .toBool()) {
		return;
	}

	// Get the current ignore list
	QStringList formats_ignored =
	    pvconfig.value(PVCONFIG_FORMATS_INVALID_IGNORED, QStringList()).toStringList();

	// And remove them from the error list
	QHash<QString, std::pair<QString, QString>> errors_ = errors;
	for (int i = 0; i < formats_ignored.size(); i++) {
		errors_.remove(formats_ignored[i]);
	}

	if (errors_.size() == 0) {
		return;
	}

	QMessageBox msg(QMessageBox::Warning, tr("Invalid formats"), tr("Some formats were invalid."));
	msg.setInformativeText(
	    tr("You can simply ignore this message, choose not to display it again "
	       "(for every format), or remove this warning only for these formats."));
	QPushButton* ignore = msg.addButton(QMessageBox::Ignore);
	msg.setDefaultButton(ignore);
	QPushButton* always_ignore =
	    msg.addButton(tr("Always ignore these formats"), QMessageBox::AcceptRole);
	QPushButton* never_again =
	    msg.addButton(tr("Never display this message again"), QMessageBox::RejectRole);

	QString detailed_txt;
	for (auto it = errors_.begin(); it != errors_.end(); ++it) {
		detailed_txt += it.value().first + QString(" (") + it.key() + QString("): ") +
		                it.value().second + QString("\n");
	}
	msg.setDetailedText(detailed_txt);

	msg.exec();

	QPushButton* clicked_btn = (QPushButton*)msg.clickedButton();

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

void PVInspector::PVMainWindow::reset_root()
{
	get_root().clear();
}

void PVInspector::PVMainWindow::close_solution()
{
	if (!maybe_save_solution()) {
		return;
	}
	close_solution_Slot();
}

std::string PVInspector::PVMainWindow::get_next_scene_name()
{
	return tr("Data collection %1").arg(sequence_n++).toStdString();
}

// Mainly from Qt's SDI example
PVInspector::PVMainWindow* PVInspector::PVMainWindow::find_main_window(const QString& path)
{
	QString canonicalFilePath = QFileInfo(path).canonicalFilePath();

	for (QWidget* widget : qApp->topLevelWidgets()) {
		PVMainWindow* mw = qobject_cast<PVMainWindow*>(widget);
		if (mw && QFileInfo(mw->get_solution_path()).canonicalFilePath() == canonicalFilePath)
			return mw;
	}
	return nullptr;
}
