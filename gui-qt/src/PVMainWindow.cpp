/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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

#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVNrawException.h>
#include <pvkernel/rush/PVUnicodeSourceError.h>

#include <pvkernel/widgets/PVColorDialog.h>

#include <inendi/PVSelection.h>
#include <inendi/PVStateMachine.h>
#include <inendi/PVSource.h>

#include <pvparallelview/PVParallelView.h>
#include <pvguiqt/PVExportSelectionDlg.h>

#include <PVFormatBuilderWidget.h>

#include <tbb/tick_count.h>

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
	pv_font_database.addApplicationFont(QString(":/Jura-DemiBold.ttf"));
	pv_font_database.addApplicationFont(QString(":/OSP-DIN.ttf"));

	// import_source = nullptr;
	report_started = false;
	report_image_index = 0;
	report_filename = nullptr;

	// We activate all available Windows
	_projects_tab_widget = new PVGuiQt::PVProjectsTabWidget(&get_root());
	_projects_tab_widget->show();
	connect(_projects_tab_widget, SIGNAL(new_project()), this, SLOT(solution_new_Slot()));
	connect(_projects_tab_widget, SIGNAL(load_project()), this, SLOT(solution_load_Slot()));
	connect(_projects_tab_widget, SIGNAL(load_project_from_path(const QString&)), this,
	        SLOT(load_solution_and_create_mw(const QString&)));
	connect(_projects_tab_widget, SIGNAL(save_project()), this, SLOT(solution_save_Slot()));
	connect(_projects_tab_widget, SIGNAL(load_source_from_description(PVRush::PVSourceDescription)),
	        this, SLOT(load_source_from_description_Slot(PVRush::PVSourceDescription)));
	connect(_projects_tab_widget, SIGNAL(import_type(const QString&)), this,
	        SLOT(import_type_Slot(const QString&)));
	connect(_projects_tab_widget, SIGNAL(new_format()), this, SLOT(new_format_Slot()));
	connect(_projects_tab_widget, SIGNAL(load_format()), this, SLOT(open_format_Slot()));
	connect(_projects_tab_widget, SIGNAL(edit_format(const QString&)), this,
	        SLOT(edit_format_Slot(const QString&)));
	connect(_projects_tab_widget, SIGNAL(is_empty()), this, SLOT(close_solution_Slot()));
	connect(_projects_tab_widget, SIGNAL(active_project(bool)), this,
	        SLOT(menu_activate_is_file_opened(bool)));

	// We display the PV Icon together with a button to import files
	pv_centralMainWidget = new QWidget();
	pv_centralMainWidget->setObjectName("pv_centralMainWidget_of_PVMainWindow");

	pv_mainLayout = new QVBoxLayout();
	pv_mainLayout->setContentsMargins(0, 0, 0, 0);

	pv_mainLayout->addWidget(_projects_tab_widget);

	/**
	 * Show warning message when no GPU accelerated device has been found
	 */
	if (not PVParallelView::common::is_gpu_accelerated()) {
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

#ifdef WITH_MINESET
	connect(this, &PVInspector::PVMainWindow::mineset_error, this,
	        &PVInspector::PVMainWindow::mineset_error_slot);
#endif

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
 * PVInspector::PVMainWindow::auto_detect_formats
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::auto_detect_formats(PVFormatDetectCtxt ctxt)
{
	// Go through the inputs
	for (auto const& input : ctxt.inputs) {
		QString in_str = input->human_name();
		ctxt.hash_input_name[in_str] = input;

		// Pre-discovery to have some sources already eliminated and
		// save the custom formats of the remaining sources
		PVRush::list_creators pre_discovered_creators;
		PVRush::hash_formats custom_formats;
		for (PVRush::PVSourceCreator_p sc : ctxt.lcr) {
			if (sc->pre_discovery(input)) {
				pre_discovered_creators.push_back(sc);
				ctxt.in_t->get_custom_formats(input, custom_formats);
			}
		}

		// Load possible formats of the remaining sources
		PVRush::hash_format_creator dis_format_creator =
		    PVRush::PVSourceCreatorFactory::get_supported_formats(pre_discovered_creators);

		// Add the custom formats
		for (auto it_cus_f = custom_formats.begin(); it_cus_f != custom_formats.end(); it_cus_f++) {
			// Save this custom format to the global formats object
			ctxt.formats.insert(it_cus_f.key(), it_cus_f.value());

			for (auto src_creator : ctxt.lcr) {
				PVRush::hash_format_creator::mapped_type v(it_cus_f.value(), src_creator);
				dis_format_creator[it_cus_f.key()] = v;

				// Save this format/creator pair to the "format_creator" object
				ctxt.format_creator[it_cus_f.key()] = v;
			}
		}

		// Try every possible format
		QHash<QString, PVCore::PVMeanValue<float>> file_types;
		tbb::tick_count dis_start = tbb::tick_count::now();

		QList<PVRush::hash_format_creator::key_type> dis_formats = dis_format_creator.keys();
		QList<PVRush::hash_format_creator::mapped_type> dis_v = dis_format_creator.values();
		bool input_exception = false;
		std::string input_exception_str;
#pragma omp parallel for
		for (int i = 0; i < dis_format_creator.size(); i++) {
			// PVRush::pair_format_creator const& pfc = itfc.value();
			PVRush::pair_format_creator const& pfc = dis_v.at(i);
			// QString const& str_format = itfc.key();
			QString const& str_format = dis_formats.at(i);
			try {
				float success_rate = PVRush::PVSourceCreatorFactory::discover_input(
				    pfc, input, &_auto_detect_cancellation);

				if (success_rate > 0) {
#pragma omp critical
					{
						PVLOG_INFO("For input %s with format %s, success rate is %0.4f\n",
						           qPrintable(in_str), qPrintable(str_format), success_rate);
						file_types[str_format].push(success_rate);
						ctxt.discovered_types[str_format].push(success_rate);
					}
				}
			} catch (PVRush::PVXmlParamParserException& e) {
#pragma omp critical
				{
					ctxt.formats_error[pfc.first.get_full_path()] = std::pair<QString, QString>(
					    pfc.first.get_format_name(), tr("XML parser error: ") + e.what());
				}
				continue;
			} catch (PVRush::PVFormatInvalid& e) {
#pragma omp critical
				{
					ctxt.formats_error[pfc.first.get_full_path()] =
					    std::pair<QString, QString>(pfc.first.get_format_name(), e.what());
				}
				continue;
			} catch (PVRush::PVInputException& e) {
#pragma omp critical
				{
					input_exception = true;
					input_exception_str = e.what();
				}
			}
		}
		tbb::tick_count dis_end = tbb::tick_count::now();
		PVLOG_INFO("Automatic format discovery took %0.4f seconds.\n",
		           (dis_end - dis_start).seconds());
		if (input_exception) {
			PVLOG_ERROR("PVInput exception: %s\n", input_exception_str.c_str());
			continue;
		}

		if (file_types.count() == 1) {
			// We got the formats that matches this input
			ctxt.discovered[file_types.keys()[0]].push_back(input);
		} else {
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
	    inendi_view->get_layer_stack().get_new_layer_name(), should_hide_layers);

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
	    inendi_view->get_layer_stack().get_new_layer_name(), should_hide_layers);
	if (!name.isEmpty()) {

		if (should_hide_layers) {
			inendi_view->hide_layers();
		}

		inendi_view->add_new_layer();
		Inendi::PVLayer& new_layer = inendi_view->get_current_layer();

		/* We set it's selection to the final selection */
		inendi_view->commit_selection_to_layer(new_layer);

		// We remove that selection from the current layer
		current_layer.get_selection().and_not(new_layer.get_selection());

		/* We need to reprocess the layer stack */
		inendi_view->update_current_layer_min_max();
		inendi_view->compute_selectable_count(new_layer);

		// do not forget to update the current layer
		inendi_view->compute_selectable_count(current_layer);

		inendi_view->process_layer_stack(inendi_view->get_real_output_selection());
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
						connect(action, SIGNAL(triggered()), this, SLOT(filter_Slot()));
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
			connect(action, SIGNAL(triggered()), this, SLOT(filter_Slot()));

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
	PVRush::list_creators lcr = PVRush::PVSourceCreatorFactory::get_by_input_type(in_t);
	PVRush::hash_format_creator format_creator =
	    PVRush::PVSourceCreatorFactory::get_supported_formats(lcr);

	PVRush::hash_formats formats, new_formats;

	for (auto itfc = format_creator.begin(); itfc != format_creator.end(); ++itfc) {
		formats[itfc.key()] = itfc.value().first;
	}

	// Create the input widget
	QString choosenFormat;
	// PVInputType::list_inputs is a QList<PVInputDescription_p>
	PVRush::PVInputType::list_inputs inputs;

	PVCore::PVArgumentList args;

	if (!in_t->createWidget(formats, new_formats, inputs, choosenFormat, args, this))
		return; // This means that the user pressed the "cancel" button

	// Add the new formats to the formats
	{
		for (auto it = new_formats.begin(); it != new_formats.end(); it++) {
			formats[it.key()] = it.value();
			for (auto src_creator : lcr) {
				PVRush::hash_format_creator::mapped_type v(it.value(), src_creator);
				// Save this format/creator pair to the "format_creator" object
				format_creator[it.key()] = v;
			}
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
	PVRush::list_creators lcr = PVRush::PVSourceCreatorFactory::get_by_input_type(in_t);

	QHash<QString, PVRush::PVInputType::list_inputs> discovered;
	QHash<QString, PVCore::PVMeanValue<float>> discovered_types; // format->mean_success_rate

	QHash<QString, std::pair<QString, QString>> formats_error; // Errors w/ some formats

	map_files_types files_multi_formats;
	QHash<QString, PVRush::PVInputDescription_p> hash_input_name;

	bool file_type_found = false;

	if (choosenFormat.compare(INENDI_AUTOMATIC_FORMAT_STR) == 0) {
		set_auto_detect_cancellation(false);

		if (!PVCore::PVProgressBox::progress(
		        [&](PVCore::PVProgressBox& pbox) {
			        pbox.set_enable_cancel(true);
			        connect(&pbox, SIGNAL(rejected()), this, SLOT(set_auto_detect_cancellation()));
			        auto_detect_formats(PVFormatDetectCtxt(
			            inputs, hash_input_name, formats, format_creator, files_multi_formats,
			            discovered, formats_error, lcr, in_t, discovered_types));
			    },
		        tr("Auto-detecting file format..."), this)) {
			return;
		}
		file_type_found = (discovered.size() > 0) | (files_multi_formats.size() > 0);
	} else if (choosenFormat.compare(INENDI_LOCAL_FORMAT_STR) == 0) {
		PVRush::hash_formats custom_formats;
		PVRush::list_creators pre_discovered_creators;

		for (auto input_it = inputs.begin(); input_it != inputs.end(); ++input_it) {
			QString in_str = (*input_it)->human_name();
			hash_input_name[in_str] = *input_it;

			for (auto cr_it = lcr.begin(); cr_it != lcr.end(); ++cr_it) {
				PVRush::PVSourceCreator_p sc = *cr_it;
				if (sc->pre_discovery(*input_it)) {
					pre_discovered_creators.push_back(sc);
					in_t->get_custom_formats(*input_it, custom_formats);
				}
			}

			for (auto hf_it = custom_formats.begin(); hf_it != custom_formats.end(); ++hf_it) {
				formats.insert(hf_it.key(), hf_it.value());

				for (auto src_cr_it = lcr.begin(); src_cr_it != lcr.end(); ++src_cr_it) {
					PVRush::hash_format_creator::mapped_type v(hf_it.value(), *src_cr_it);
					format_creator[hf_it.key()] = v;
				}
			}
		}

		if (custom_formats.size() == 1) {
			file_type_found = true;
			discovered[custom_formats.keys()[0]] = inputs;
		}
	} else if (choosenFormat.compare(INENDI_BROWSE_FORMAT_STR) == 0) {
		/* A QFileDialog is explicitly used over QFileDialog::getOpenFileName
		 * because this latter does not used QFileDialog's global environment
		 * like last used current directory.
		 */
		QFileDialog* fdialog = new QFileDialog(this);

		fdialog->setOption(QFileDialog::DontUseNativeDialog, true);
		fdialog->setNameFilter("Formats (*.format)");
		fdialog->setWindowTitle("Load format from...");

		int ret = fdialog->exec();

		if (ret == QDialog::Accepted) {
			QString format_path = fdialog->selectedFiles().at(0);
			QFileInfo fi(format_path);
			QString format_name = "custom:" + fi.dir().path();

			PVRush::PVFormat format(format_name, format_path);
			formats[format_name] = format;

			for (auto src_cr_it = lcr.begin(); src_cr_it != lcr.end(); ++src_cr_it) {
				PVRush::hash_format_creator::mapped_type v(format, *src_cr_it);
				format_creator[format_name] = v;
			}

			if (fi.isReadable()) {
				file_type_found = true;
				discovered[format_name] = inputs;
			}
		}

		delete fdialog;

		if (ret == QDialog::Rejected) {
			return;
		}
	} else {
		file_type_found = true;
		discovered[choosenFormat] = inputs;
	}

	treat_invalid_formats(formats_error);

	if (!file_type_found) {
		QString msg;
		if (choosenFormat.compare(INENDI_AUTOMATIC_FORMAT_STR) == 0) {
			msg = "<p>Automatic format detection reported <strong>no valid "
			      "format</strong>.</p>";
			msg += "<p>Please note that automatic format detection is only applied "
			       "on a small subset of the provided sources.</p>";
			msg += "<p><strong>Trick:</strong> if you know the format of these "
			       "sources, and if it contains one or more filters that invalidate "
			       "a lot of elements, you should avoid automatic format detection "
			       "and select this format by hand in the import sources dialog.</p>";
		} else if (choosenFormat.compare(INENDI_BROWSE_FORMAT_STR) == 0) {
			msg = "<p>No valid format file found.</p>";
			msg += "<p>Check for file permission on the chosen format file.</p>";
		} else if (choosenFormat.compare(INENDI_LOCAL_FORMAT_STR) == 0) {
			// must never happens
			msg = "<p>No valid local format file found.</p>";
			msg += "<ul>";
			msg += "<li>the source's directory contains a readable format file named "
			       "<em>inendi.format</em> (or <em>picviz.format</em> for backward "
			       "compatibility)</li>";
			msg += "<li>the source file has a format file whose name is "
			       "<em>file.ext<strong>.format</strong></em></li>";
			msg += "</ul>";
		}
		QMessageBox::warning(this, "Cannot import sources", msg);
		return;
	}

	if (discovered_types.size() > 1) {
		QStringList dis_types = discovered_types.keys();
		QStringList dis_types_comment;
		QList<PVCore::PVMeanValue<float>> rates = discovered_types.values();
		for (PVCore::PVMeanValue<float> const& mean : rates) {
			dis_types_comment << QString("mean success rate = %1%").arg(mean.compute_mean() * 100);
		}

		PVStringListChooserWidget* choosew = new PVStringListChooserWidget(
		    this, "Multiple types have been detected.\nPlease choose the one(s) "
		          "you need and press OK.",
		    dis_types, dis_types_comment);
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
				++it;
				files_multi_formats.erase(it_rem);
			} else {
				++it;
			}
		}
	}

	if (files_multi_formats.size() > 0) {
		PVFilesTypesSelWidget* files_types_sel =
		    new PVFilesTypesSelWidget(this, files_multi_formats);
		if (!files_types_sel->exec())
			return;
		// Add everything to the discovered table
		for (auto const& file_types : files_multi_formats) {
			QStringList const& types_l = file_types.second;
			QString const& input_name = file_types.first;
			for (int i = 0; i < types_l.size(); i++) {
				discovered[types_l[i]] << hash_input_name[input_name];
			}
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

		PVRush::pair_format_creator const& fc = format_creator[it.key()];

		PVRush::PVFormat const& cur_format = fc.first;

		PVRush::PVSourceDescription src_desc(inputs, fc.second, cur_format);

		try {
			if (load_source_from_description_Slot(src_desc)) {
				one_extraction_successful = true;
			}
		} catch (Inendi::InvalidPlottingMapping const& e) {
			invalid_formats.append(it.key() + ": " + e.what());
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
	PVRush::list_creators lcr = PVRush::PVSourceCreatorFactory::get_by_input_type(in_file);
	PVRush::hash_format_creator format_creator =
	    PVRush::PVSourceCreatorFactory::get_supported_formats(lcr);

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
			files_in.push_back(
			    PVRush::PVInputDescription_p(new PVRush::PVFileDescription(filename)));
		}
	}

	if (!format.isEmpty()) {
		PVRush::PVFormat new_format("custom:arg", format);
		formats["custom:arg"] = new_format;

		for (auto src_creator : lcr) {
			PVRush::hash_format_creator::mapped_type v(new_format, src_creator);
			// Save this format/creator pair to the "format_creator" object
			format_creator["custom:arg"] = v;
		}
		format = "custom:arg";
	} else {
		format = INENDI_AUTOMATIC_FORMAT_STR;
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

	QString img_name =
	    QFileDialog::getSaveFileName(this, title, filename, QString("PNG Image (*.png)"));

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

static void update_status_ext(PVCore::PVProgressBox& pbox, PVRush::PVControllerJob_p job)
{
	while (job->running()) {
		pbox.set_status(job->status());
		pbox.set_extended_status(
		    QString("Number of rejected elements: %L1").arg(job->rejected_elements()));
		boost::this_thread::sleep(boost::posix_time::milliseconds(200));
	}
}

static QString bad_conversions_as_string(
    const PVRush::PVNraw::unconvertable_values_t::bad_conversions_t& bad_conversions,
    const Inendi::PVSource* src)
{
	QStringList l;

	auto const& ax = src->get_format().get_axes();

	size_t max_values = 1000;

	for (const auto& bad_conversion : bad_conversions) {

		const PVRow row = bad_conversion.first;
		QString str("row #" + QString::number(row + 1) + " :");

		for (const auto& bad_field : bad_conversion.second) {
			const PVCol col = bad_field.first;
			const QString& axis_name = ax[col].get_name();
			const QString& axis_type = ax[col].get_type();

			str += " " + axis_name + " (" + axis_type + ") : \"" +
			       QString::fromStdString(bad_field.second) + "\"";
		}

		l << str;
		if (max_values-- == 0) {
			l << "There are more errors but we only show first 1000 errors. Fix you format type!";
			break;
		}
	}

	return l.join("\n");
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::load_source
 *
 *****************************************************************************/
bool PVInspector::PVMainWindow::load_source(Inendi::PVSource* src)
{
	// Load a created source
	// Extract the source
	BENCH_START(lff);

	// PVCore::PVProgressBox::progress();
	PVRush::PVControllerJob_p job_import;
	try {
		job_import = src->extract(0);
	} catch (PVRush::PVInputException const& e) {
		QMessageBox::critical(this, "Cannot create sources",
		                      QString("Error with input: ") + e.what());
		return false;
	} catch (PVRush::PVNrawException const& e) {
		QMessageBox::critical(this, "Cannot create sources",
		                      QString("Error with nraw: ") + e.what());
		return false;
	}

	bool ret = PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& pbox) {

		    pbox.set_detail_label(QString("Number of elements extracted: %L1"));
		    pbox.set_cancel2_btn_text("Stop and process");
		    pbox.set_cancel_btn_text("Discard");
		    pbox.set_confirmation(true);
		    QProgressBar* pbar = pbox.getProgressBar();
		    pbar->setValue(0);
		    // set min and max to 0 to have an activity effect
		    // FIXME : We should be able to use nlines as max.
		    pbar->setMaximum(job_import->nb_elts_max());
		    pbar->setMinimum(0);

		    QObject::connect(job_import.get(), SIGNAL(job_done_signal()), &pbox, SLOT(accept()));
		    // launch a thread in order to update the status of the progress bar
		    boost::thread th_status([&]() { update_status_ext(pbox, job_import); });
		    pbox.launch_timer_status();

		    // Show the progressBox
		    if (job_import->done() or pbox.exec() == QDialog::Accepted) {
			    // Job finished, everything is fine.
			    return true;
		    }

		    // Cancel this job and ask the user if he wants to keep the extracted data.
		    job_import->cancel();
		    PVLOG_DEBUG("extractor: job canceled !\n");
		    // Sucess if we ask to continue with loaded data.
		    return (pbox.get_cancel_state() == PVCore::PVProgressBox::CANCEL2);

		},
	    QString("Extracting %1...").arg(src->get_format_name()), this);

	if (not ret) {
		// If job is canceled, stop here
		return false;
	}
	try {
		src->wait_extract_end(job_import);
	} catch (PVRush::PVInputException const& e) {
		QMessageBox::critical(this, "Cannot create sources",
		                      QString("Error with input: ") + e.what());
		return false;
	} catch (PVRush::UnicodeSourceError const&) {
		QMessageBox::critical(this, "Cannot create sources",
		                      "File encoding does permit Inspector to perform extraction.");
		return false;
	}

	if (src->get_rushnraw().get_row_count() == 0) {
		QString msg = QString("<p>The files <strong>%1</strong> using format "
		                      "<strong>%2</strong> cannot be opened. ")
		                  .arg(QString::fromStdString(src->get_name()))
		                  .arg(src->get_format_name());
		PVRow nelts = job_import->rejected_elements();
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
	} else if (size_t bc_count = src->get_rushnraw().unconvertable_values().bad_conversions_count) {
		// We can continue with it but user have to know that some values are
		// incorrect.
		QMessageBox warning_message(QMessageBox::Warning, "Failed conversion(s)",
		                            "\n" + QString::number(bc_count) +
		                                " conversions from text to binary failed during import...",
		                            QMessageBox::Ok, this);
		warning_message.setInformativeText("Such values are displayed in italic in the "
		                                   "listing, but are treated as default values "
		                                   "elsewhere.");
		warning_message.setDetailedText(bad_conversions_as_string(
		    src->get_rushnraw().unconvertable_values().bad_conversions(), src));
		warning_message.exec();
	}

	BENCH_STOP(lff);
#ifdef INENDI_DEVELOPER_MODE
	PVLOG_INFO("nraw created from data in %g sec\n", BENCH_END_TIME(lff));
#endif

	if (!PVCore::PVProgressBox::progress(
	        [&](PVCore::PVProgressBox& /*pbox*/) {
		        auto& mapped = src->emplace_add_child();
		        auto& plotted = mapped.emplace_add_child();
		        plotted.emplace_add_child();
		    },
	        tr("Processing..."), (QWidget*)this)) {
		return false;
	}

	source_loaded(*src);

	return true;
}

void PVInspector::PVMainWindow::source_loaded(Inendi::PVSource& src)
{
	// Create workspace for this source.
	_projects_tab_widget->add_source(&src);

	// Show invalide elements.
	if (src.get_invalid_evts().size() > 0) {
		display_inv_elts();
	}

	// Add format as recent format
	PVCore::PVRecentItemsManager::get().add(src.get_format().get_full_path(),
	                                        PVCore::PVRecentItemsManager::Category::USED_FORMATS);

	// Add source as recent source
	PVCore::PVRecentItemsManager::get().add_source(src.get_source_creator(), src.get_inputs(),
	                                               src.get_format());
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
