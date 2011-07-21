//! \file PVMainWindowSlots.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <PVMainWindow.h>
#include <PVArgumentListWidget.h>
#include <PVXmlEditorWidget.h>
#include "PVLayerFilterProcessWidget.h"

#include <PVCheckableComboBox.h>

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
		QString content = "Picviz Inspector\n(c) 2010-2011 Picviz Labs SAS\ncontact@picviz.com\nhttp://www.picviz.com\n\nWith CUDA support\nQT version " + QString(QT_VERSION_STR);
#else
		QString content = "Picviz Inspector\n(c) 2010-2011 Picviz Labs SAS\ncontact@picviz.com\nhttp://www.picviz.com\n\nQT version " + QString(QT_VERSION_STR);
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
	pv_AxisProperties->create();
	pv_AxisProperties->show();
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
	PVGL::PVMessage message;

	message.function = PVGL_COM_FUNCTION_TOGGLE_DISPLAY_EDGES;
	pvgl_com->post_message_to_gl(message);
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
	update_pvglview(current_lib_view, PVGL_COM_REFRESH_SELECTION);

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
	update_pvglview(current_lib_view, PVGL_COM_REFRESH_SELECTION);
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
	update_pvglview(current_lib_view, PVGL_COM_REFRESH_SELECTION);

	if (!lines_display_zombies_GLview_Action->text().compare(QString(tr("Hide zombies lines in view")))) {
		lines_display_zombies_GLview_Action->setText(QString(tr("Display zombies lines in view")));
	} else {
		lines_display_zombies_GLview_Action->setText(QString(tr("Hide zombies lines in view")));
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

	int result;

	result = pv_ExportSelectionDialog->exec();

	// if (result) {
	// 	QString filename = 
	// }
	// QString filename = pv_ExportSelectionDialog->getSaveFileName();
	// if (!filename.isEmpty()) {
	// 	PVLOG_INFO("The user pressed OK and choosed the file %s\n", qPrintable(filename));
	// }

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
		Picviz::PVLayerFilter::p_type filter_org = LIB_FILTER(Picviz::PVLayerFilter)::get().get_filter_by_name(filter_name);
		//cpy filter
		Picviz::PVLayerFilter::p_type fclone = filter_org->clone<Picviz::PVLayerFilter>();
		PVCore::PVArgumentList &args = lib_view->filters_args[filter_name];
		PVLayerFilterProcessWidget* filter_widget = new PVLayerFilterProcessWidget(current_tab, args, fclone);
		filter_widget->init();
		filter_widget->show();

#if 0
		if (args.size() > 0) {
			//view the widget to param filter

			PVArgumentListWidget* arg_widget = new PVArgumentListWidget(*lib_view, args, this);
			if (!arg_widget->exec())
				return;
			lib_view->filters_args[filter_name] = args;
			fclone->set_args(args);

		}

		// Then...

		lib_view->process_selection();

		fclone->set_view(lib_view);
		fclone->set_output(&lib_view->post_filter_layer);
		PVLOG_INFO("in    %s l:%d\n",__FUNCTION__,__LINE__);
		
		//call filter in a thread and view the progress dialog
		PVProgressBox *progressDialog = new PVProgressBox(tr("Applying filter..."), this, 0);
		QFuture<void> worker = QtConcurrent::run<void>(process_layer_filter, fclone.get(), &lib_view->pre_filter_layer);
		QFutureWatcher<void> watcher;
		watcher.setFuture(worker);
		QObject::connect(&watcher, SIGNAL(finished()), progressDialog, SLOT(accept()), Qt::QueuedConnection);
		
		if(progressDialog->exec()){//not cancelled...
			PVLOG_DEBUG("Filtering action performed\n");
			lib_view->pre_filter_layer = lib_view->post_filter_layer;
			lib_view->state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);

			// We reprocess the pipeline from the eventline stage
			lib_view->process_from_eventline();
			update_pvglview(current_tab->get_lib_view(), PVGL_COM_REFRESH_SELECTION|PVGL_COM_REFRESH_COLOR);
			current_tab->refresh_listing_Slot();
		}else{//if it's cancelled...
			PVLOG_DEBUG("Filtering action canceled\n");
			disconnect(&watcher,  SIGNAL(finished()),0,0);
			worker.cancel();
			lib_view->post_filter_layer = lib_view->pre_filter_layer;
		}

		
		/* THEN we can emit the signal */
		// commit_selection_in_current_layer(current_tab->get_lib_view());
		emit filter_applied_Signal();
#endif
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

/******************************************************************************
 *
 * PVInspector::PVMainWindow::new_file_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::new_file_Slot()
{
	// nothing here yet
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::new_scene_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::new_scene_Slot()
{
	// nothing here yet
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
 * PVInspector::PVMainWindow::remove_log_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::remote_log_Slot()
{
#if 0
	qDebug("PVInspector::PVMainWindow::%s\n", __FUNCTION__);

	QMenuBar *rl_menuBar = new QMenuBar(0);
	QMenu *rl_fileMenu = rl_menuBar->addMenu( tr( "Machine" ) );
	// QDialogButtonBox *buttons = new QDialogButtonBox(Qt::Vertical);

	QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok
					 | QDialogButtonBox::Cancel);

	QWidget *rl_central = new QWidget(this);
	QVBoxLayout *rl_layout = new QVBoxLayout;

	rl_fileMenu->addAction(pv_RemoteLog->addMachineAction());
	rl_fileMenu->addAction(pv_RemoteLog->removeMachineAction());

	RemoteLogDialog->setMenuBar(rl_menuBar);
	rl_menuBar->show();

	rl_central->setLayout(rl_layout);
	rl_layout->addWidget(pv_RemoteLog);
	rl_layout->addWidget(buttonBox);

	// connect(buttonBox, SIGNAL(accepted()), this, SLOT(accept()));
	connect(buttonBox, SIGNAL(rejected()), RemoteLogDialog, SLOT(hide()));

	RemoteLogDialog->setWindowTitle(tr("Import remote file"));
	RemoteLogDialog->setCentralWidget(rl_central);

	RemoteLogDialog->show();
#endif
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

	if (current_tab && current_tab->get_lib_view()) {
		Picviz::PVView_p view = current_tab->get_lib_view();
		view->select_all_nonzb_lines();
		update_pvglview(view, PVGL_COM_REFRESH_SELECTION);
		current_tab->refresh_listing_Slot();
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

	if (current_tab && current_tab->get_lib_view()) {
// picviz_selection_A2A_inverse(current_tab->get_lib_view()->volatile_selection);
		current_tab->get_lib_view()->volatile_selection.select_none();
		current_tab->get_lib_view()->process_from_selection();
		current_tab->get_lib_view()->process_from_eventline();
		update_pvglview(current_tab->get_lib_view(), PVGL_COM_REFRESH_SELECTION);
		current_tab->refresh_listing_Slot();
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

	if (current_tab && current_tab->get_lib_view()) {
// picviz_selection_A2A_inverse(current_tab->get_lib_view()->volatile_selection);
		current_tab->get_lib_view()->volatile_selection = ~(current_tab->get_lib_view()->volatile_selection);
		current_tab->get_lib_view()->process_from_selection();
		current_tab->get_lib_view()->process_from_eventline();
		update_pvglview(current_tab->get_lib_view(), PVGL_COM_REFRESH_SELECTION);
		current_tab->refresh_listing_Slot();
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
	float lib_ver;

	QMessageBox msgBox;

	QByteArray data=reply->readAll();
	QString str(data);
	float http_ver = str.toFloat();

	lib_ver = atof(PICVIZ_VERSION_STR);

	if (http_ver > lib_ver) {
		PVLOG_INFO("You are running version %s, there is a new version available: %s\n", PICVIZ_VERSION_STR, qPrintable(str));
		msgBox.information(this, "New version available", "There is a new version available.\n Please contact Picviz Labs (contact@picviz.com) to get an update");
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
		PVGL::PVMessage message;

		message.function = PVGL_COM_FUNCTION_CREATE_SCATTER_VIEW;
		message.pv_view = current_tab->get_lib_view();
		message.pointer_1 = new QString(pv_ListingsTabWidget->tabText(pv_ListingsTabWidget->currentIndex()));
		pvgl_com->post_message_to_gl(message);
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::view_open_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::view_open_Slot()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);
	QStringList file_type_and_file_name;
	QString file_name, file_type, file_absolute_path;
	QMessageBox msgBox;

	if (current_tab && current_tab->get_lib_view()) {
		file_type_and_file_name = pv_OpenFileDialog->getFileName();
		file_absolute_path = file_type_and_file_name[0];

		if ( ! file_absolute_path.isEmpty() ) {
			file_name = file_absolute_path.split("/").takeLast();
		} else {
			return;
		}

/* FIXME		if (!picviz_open_is_picviz_type(file_absolute_path.toUtf8().data())) {
			msgBox.critical(this, "Not a Picviz file", "This file is not a Picviz file. Cannot open it!");
			return;
		}

		picviz_open_inline(current_tab->get_lib_view(), file_absolute_path.toUtf8().data());
*/
		update_pvglview(current_tab->get_lib_view(), PVGL_COM_REFRESH_POSITIONS|PVGL_COM_REFRESH_Z|PVGL_COM_REFRESH_COLOR|PVGL_COM_REFRESH_ZOMBIES|PVGL_COM_REFRESH_SELECTION);
		current_tab->refresh();
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::view_save_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::view_save_Slot()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);
	QStringList file_type_and_file_name;
	QString file_name, file_type, file_absolute_path;

	if (current_tab && current_tab->get_lib_view()) {
		file_type_and_file_name = pv_SaveFileDialog->getFileName();
		file_absolute_path = file_type_and_file_name[0];

		if ( ! file_absolute_path.isEmpty() ) {
			file_name = file_absolute_path.split("/").takeLast();
		} else {
			return;
		}

// FIXME		picviz_save(current_tab->get_lib_view(), file_absolute_path.toUtf8().data());
	}
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::view_show_new_Slot
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::view_show_new_Slot()
{
	// Ask the PVGL to create a GL-View of the currently selected view.
	if (current_tab && current_tab->get_lib_view()) {
		PVGL::PVMessage message;
		message.function = PVGL_COM_FUNCTION_PLEASE_WAIT;
		message.pointer_1 = new QString(pv_ListingsTabWidget->tabText(pv_ListingsTabWidget->currentIndex()));
		pvgl_com->post_message_to_gl(message);

		message.function = PVGL_COM_FUNCTION_CREATE_VIEW;
		message.pv_view = current_tab->get_lib_view();
		message.pointer_1 = new QString(pv_ListingsTabWidget->tabText(pv_ListingsTabWidget->currentIndex()));
		pvgl_com->post_message_to_gl(message);
	}
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
 * PVInspector::PVMainWindow::file_format_builder_Slot()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::file_format_builder_Slot() {
    QDialog *editorWin=new QDialog();

    PVXmlEditorWidget *editorWidget = new PVXmlEditorWidget();
    QVBoxLayout *layout = new QVBoxLayout();
    layout->setMargin(0);
    editorWin->setLayout(layout);
    editorWin->layout()->addWidget(editorWidget);
    editorWin->resize(700,500);
    editorWin->move(50,50);
    editorWin->exec();
}

PVGL::PVCom* PVInspector::PVMainWindow::get_pvcom()
{
	return pvgl_com;
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
