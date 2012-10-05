/**
 * \file PVMainWindowMenusActions.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <PVMainWindow.h>
#include <PVInputTypeMenuEntries.h>

#include <QAction>
#include <QMenuBar>

/******************************************************************************
 *
 * PVInspector::PVMainWindow::create_actions
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::create_actions()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);
	/************************
	 * For the "File" menu entry
	 ************************/

#ifdef CUSTOMER_CAPABILITY_SAVE
	// The project actions
	project_new_Action = new QAction(tr("&New project"), this);
	project_load_Action = new QAction(tr("&Load a project..."), this);
	project_save_Action = new QAction(tr("&Save project"), this);
	project_saveas_Action = new QAction(tr("S&ave project as..."), this);
#endif	// CUSTOMER_CAPABILITY_SAVE

	// The new_file Action
	new_file_Action = new QAction(tr("&New"), this);
	new_file_Action->setIcon(QIcon(":/document-new.png"));
	new_file_Action->setShortcut(QKeySequence::New);
	new_file_Action->setStatusTip(tr("Create a new file."));
	new_file_Action->setWhatsThis(tr("Use this to create a new file."));

	// Export our selection Action
	export_selection_Action = new QAction(tr("Export &selection..."), this);
	export_selection_Action->setToolTip(tr("Export our current selection"));

	// The extractorFile Action
	extractor_file_Action = new QAction(tr("&Extractor..."), this);
	extractor_file_Action->setToolTip(tr("Launch the Picviz Extractor"));
	extractor_file_Action->setEnabled(false);

	export_file_Action = new QAction(tr("&Export"), this);

	quit_Action = new QAction(tr("&Quit"), this);



	/************************
	 * For the "Edit" menu entry
	 ************************/

	undo_Action = new QAction(tr("Undo"), this);
	undo_Action->setIcon(QIcon(":/edit-undo.png"));

	redo_Action = new QAction(tr("Redo"), this);
	redo_Action->setIcon(QIcon(":/edit-redo.png"));

	undo_history_Action = new QAction(tr("Undo history"), this);

	cut_Action = new QAction(tr("Cut"), this);
	cut_Action->setIcon(QIcon(":/edit-cut.png"));

	copy_Action = new QAction(tr("Copy"), this);
	copy_Action->setIcon(QIcon(":/edit-copy.png"));

	paste_Action = new QAction(tr("Paste"), this);
	paste_Action->setIcon(QIcon(":/edit-paste.png"));


	/************************
	 * For the "Selection" menu entry
	 ************************/
	selection_all_Action = new QAction(tr("&All"), this);
	selection_all_Action->setShortcut(QKeySequence(Qt::Key_A));
	selection_none_Action = new QAction(tr("&None"), this);
	selection_inverse_Action = new QAction(tr("&Inverse"), this);
	selection_inverse_Action->setShortcut(QKeySequence(Qt::Key_I));
	set_color_Action = new QAction(tr("Set color"), this);
	set_color_Action->setShortcut(QKeySequence(Qt::Key_C));
	selection_from_current_layer_Action = new QAction(tr("Set selection from current layer"), this);
	selection_from_layer_Action = new QAction(tr("Set selection from layer..."), this);

	//commit_selection_in_current_layer_Action = new QAction(tr("Keep &current layer"), this);
	//commit_selection_in_current_layer_Action->setShortcut(QKeySequence(Qt::Key_K));
	commit_selection_to_new_layer_Action = new QAction(tr("Create new layer from selection"), this);
	commit_selection_to_new_layer_Action->setShortcut(QKeySequence(Qt::ALT + Qt::Key_K));
	move_selection_to_new_layer_Action = new QAction(tr("Move selection to new layer"), this);
	move_selection_to_new_layer_Action->setShortcut(QKeySequence(Qt::ALT + Qt::Key_M));
	expand_selection_on_axis_Action = new QAction(tr("Expand selection on axis..."), this);

	/******************************
	 * For the "Filter" menu entry
	 ******************************/
	filter_reprocess_last_filter = new QAction(tr("Apply last filter..."), this);
	filter_reprocess_last_filter->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_F));

	/************************
	 * For the "Scene" menu entry
	 ************************/
	scene_menu_event_filter = new SceneMenuEventFilter(this);
	new_scene_Action = new QAction(tr("&New Scene"), this);
	select_scene_Action = new QAction(tr("&Select Scene"), this);
	correlation_scene_Action = new QAction(tr("&Correlations..."), this);

	/************************
	 * For the "Tools" menu entry
	 ************************/
	tools_new_format_Action = new QAction(tr("&New format..."), this);
	tools_cur_format_Action = new QAction(tr("&Edit current format..."), this);

	/************************
	 * For the "View" menu entry
	 ************************/
	view_new_parallel_Action = new QAction(tr("New &parallel view"), this);
	view_new_zoomed_parallel_Action = new QAction(tr("New &zoomed parallel view"), this);
	view_new_scatter_Action = new QAction(tr("New scatter &view"), this);
	view_display_inv_elts_Action = new QAction(tr("&Display invalid elements..."), this);
#ifndef NDEBUG
	view_screenshot_qt = new QAction(tr("Display view in Qt"), this);
	view_screenshot_qt->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_T));
#endif

	/***************************
	 * For the "Axes" menu entry
	 ***************************/
	axes_editor_Action = new QAction(tr("Edit axes..."), this);
	axes_combination_editor_Action = new QAction(tr("Edit axes combination..."), this);
	axes_mode_Action = new QAction(tr("Enter axes mode"), this);
	axes_mode_Action->setShortcut(QKeySequence(Qt::Key_X));
	axes_display_edges_Action = new QAction(tr("Display edges"), this);
	axes_display_edges_Action->setShortcut(QKeySequence(Qt::Key_Y));
	axes_new_Action = new QAction(tr("Create new axis..."), this);

	/***************************
	 * For the "Lines" menu entry
	 ***************************/
	lines_display_unselected_GLview_Action = new QAction(tr("Toggle unselected lines"), this);
	lines_display_unselected_GLview_Action->setShortcut(QKeySequence(Qt::Key_U));
	lines_display_unselected_listing_Action = new QAction(tr("Toggle unselected lines in listing"), this);
	lines_display_unselected_listing_Action->setShortcut(QKeySequence(Qt::SHIFT + Qt::Key_U));

	lines_display_zombies_GLview_Action = new QAction(tr("Toggle zombies lines"), this);
	lines_display_zombies_GLview_Action->setShortcut(QKeySequence(Qt::Key_Z));
	lines_display_zombies_listing_Action = new QAction(tr("Toggle zombies lines in listing"), this);
	lines_display_zombies_listing_Action->setShortcut(QKeySequence(Qt::SHIFT + Qt::Key_Z));

	/**************************
	 * For the "Help" menu entry
	 **************************/
	about_Action = new QAction(tr("&About"), this);
	//whats_this_Action = new QAction(tr("&What's this?"), this);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::create_menus
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::create_menus()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);

	menubar = menuBar();

	file_Menu = menubar->addMenu(tr("&File"));
#ifdef CUSTOMER_CAPABILITY_SAVE
	QMenu *project_Menu = new QMenu(tr("&Project"));
	project_Menu->addAction(project_new_Action);
	project_Menu->addAction(project_load_Action);
	project_Menu->addAction(project_save_Action);
	project_Menu->addAction(project_saveas_Action);

	file_Menu->addMenu(project_Menu);
	file_Menu->addSeparator();
	file_Menu->addAction(extractor_file_Action);
	file_Menu->addSeparator();
#endif
	file_Menu->addSeparator();
	QMenu *import_Menu = new QMenu(tr("I&mport"));
	create_actions_import_types(import_Menu);
	file_Menu->addMenu(import_Menu);
	QMenu *export_Menu = new QMenu(tr("E&xport"));
	export_Menu->addAction(export_selection_Action);
	file_Menu->addMenu(export_Menu);
	file_Menu->addSeparator();
	file_Menu->addAction(quit_Action);

	//edit_Menu = menubar->addMenu(tr("&Edit"));
	//edit_Menu->addAction(undo_Action);
	//edit_Menu->addAction(redo_Action);
	//edit_Menu->addAction(undo_history_Action);
	//edit_Menu->addSeparator();
	//edit_Menu->addAction(cut_Action);
	//edit_Menu->addAction(copy_Action);
	//edit_Menu->addAction(paste_Action);
	//edit_Menu->addSeparator();


	selection_Menu = menubar->addMenu(tr("&Selection"));
	selection_Menu->addAction(selection_all_Action);
	selection_Menu->addAction(selection_none_Action);
	selection_Menu->addAction(selection_inverse_Action);
	selection_Menu->addSeparator();
	selection_Menu->addAction(selection_from_current_layer_Action);
	selection_Menu->addAction(selection_from_layer_Action);
	selection_Menu->addSeparator();
	selection_Menu->addAction(set_color_Action);
	selection_Menu->addSeparator();
	//selection_Menu->addAction(commit_selection_in_current_layer_Action);
	selection_Menu->addAction(commit_selection_to_new_layer_Action);
	selection_Menu->addAction(move_selection_to_new_layer_Action);
	selection_Menu->addSeparator();
	selection_Menu->addAction(expand_selection_on_axis_Action);

	filter_Menu = menubar->addMenu(tr("Fil&ters"));
	filter_Menu->addAction(filter_reprocess_last_filter);
	filter_Menu->addSeparator();
	create_filters_menu_and_actions();

	// layer_Menu = menubar->addMenu(tr("&Layers"));
	// layer_Menu->addAction(commit_selection_in_current_layer_Action);
	// layer_Menu->addAction(commit_selection_to_new_layer_Action);

	scene_Menu = menubar->addMenu(tr("S&cene"));
	//scene_Menu->addAction(new_scene_Action);
	//scene_Menu->addAction(select_scene_Action);
	scene_Menu->addAction(correlation_scene_Action);
	
	view_Menu = menubar->addMenu(tr("&View"));
	//view_Menu->addAction(view_new_parallel_Action);
	//view_Menu->addAction(view_new_zoomed_parallel_Action);
	//view_Menu->addAction(view_new_scatter_Action);
	//view_Menu->addSeparator();
	view_Menu->addAction(view_display_inv_elts_Action);
#ifndef NDEBUG
	view_Menu->addSeparator();
	view_Menu->addAction(view_screenshot_qt);
#endif

	axes_Menu = menubar->addMenu(tr("&Axes"));
	axes_Menu->addAction(axes_editor_Action);
	axes_Menu->addAction(axes_combination_editor_Action);
	axes_Menu->addSeparator();
	axes_Menu->addAction(axes_mode_Action);
	axes_Menu->addAction(axes_display_edges_Action);
	axes_Menu->addAction(axes_new_Action);
	axes_Menu->addSeparator();


	lines_Menu = menubar->addMenu(tr("&Lines"));
	lines_Menu->addAction(lines_display_unselected_listing_Action);
	lines_Menu->addAction(lines_display_unselected_GLview_Action);
	lines_Menu->addSeparator();
	lines_Menu->addAction(lines_display_zombies_listing_Action);
	lines_Menu->addAction(lines_display_zombies_GLview_Action);

	tools_Menu = menubar->addMenu(tr("T&ools"));
	tools_Menu->addAction(tools_new_format_Action);
	tools_Menu->addAction(tools_cur_format_Action);

	windows_Menu = menubar->addMenu(tr("&Windows"));

	help_Menu = menubar->addMenu(tr("&Help"));
	//help_Menu->addAction(whats_this_Action);
	help_Menu->addAction(about_Action);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::create_actions_import_types
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::create_actions_import_types(QMenu* menu)
{
	PVInputTypeMenuEntries::add_inputs_to_menu(menu, this, SLOT(import_type_Slot()));
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::menu_activate_is_file_opened
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::menu_activate_is_file_opened(bool cond)
{
	extractor_file_Action->setEnabled(false);
	export_selection_Action->setEnabled(cond);

	axes_Menu->setEnabled(cond);
	filter_Menu->setEnabled(cond);
	lines_Menu->setEnabled(cond);
	scene_Menu->setEnabled(cond);
	selection_Menu->setEnabled(cond);
	tools_cur_format_Action->setEnabled(cond);
	view_Menu->setEnabled(cond);
	windows_Menu->setEnabled(cond);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::connect_actions()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::connect_actions()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);
#ifdef CUSTOMER_CAPABILITY_SAVE
	connect(project_new_Action, SIGNAL(triggered()), this, SLOT(project_new_Slot()));
	connect(project_load_Action, SIGNAL(triggered()), this, SLOT(project_load_Slot()));
	connect(project_save_Action, SIGNAL(triggered()), this, SLOT(project_save_Slot()));
	connect(project_saveas_Action, SIGNAL(triggered()), this, SLOT(project_saveas_Slot()));
#endif
	connect(export_file_Action, SIGNAL(triggered()), this, SLOT(export_file_Slot()));
	connect(export_selection_Action, SIGNAL(triggered()), this, SLOT(export_selection_Slot()));
	connect(extractor_file_Action, SIGNAL(triggered()), this, SLOT(extractor_file_Slot()));
	connect(quit_Action, SIGNAL(triggered()), this, SLOT(quit_Slot()));

	connect(view_new_parallel_Action, SIGNAL(triggered()), this, SLOT(view_new_parallel_Slot()));
	connect(view_new_zoomed_parallel_Action, SIGNAL(triggered()), this, SLOT(view_new_zoomed_parallel_Slot()));
	connect(view_new_scatter_Action, SIGNAL(triggered()), this, SLOT(view_new_scatter_Slot()));
	connect(view_display_inv_elts_Action, SIGNAL(triggered()), this, SLOT(view_display_inv_elts_Slot()));
#ifndef NDEBUG
	connect(view_screenshot_qt, SIGNAL(triggered()), this, SLOT(view_screenshot_qt_Slot()));
#endif

	connect(selection_all_Action, SIGNAL(triggered()), this, SLOT(selection_all_Slot()));
	connect(selection_none_Action, SIGNAL(triggered()), this, SLOT(selection_none_Slot()));
	connect(selection_inverse_Action, SIGNAL(triggered()), this, SLOT(selection_inverse_Slot()));
	connect(selection_from_current_layer_Action, SIGNAL(triggered()), this, SLOT(selection_set_from_current_layer_Slot()));
	connect(selection_from_layer_Action, SIGNAL(triggered()), this, SLOT(selection_set_from_layer_Slot()));
	connect(expand_selection_on_axis_Action, SIGNAL(triggered()), this, SLOT(expand_selection_on_axis_Slot()));

	connect(set_color_Action, SIGNAL(triggered()), this, SLOT(set_color_Slot()));

	scene_Menu->installEventFilter(scene_menu_event_filter);
	connect(correlation_scene_Action, SIGNAL(triggered()), this, SLOT(show_correlation_Slot()));

	//connect(commit_selection_in_current_layer_Action, SIGNAL(triggered()), this, SLOT(commit_selection_in_current_layer_Slot()));
	connect(commit_selection_to_new_layer_Action, SIGNAL(triggered()), this, SLOT(commit_selection_to_new_layer_Slot()));
	connect(move_selection_to_new_layer_Action, SIGNAL(triggered()), this, SLOT(move_selection_to_new_layer_Slot()));

	connect(axes_editor_Action, SIGNAL(triggered()), this, SLOT(axes_editor_Slot()));//
	connect(axes_combination_editor_Action, SIGNAL(triggered()), this, SLOT(axes_combination_editor_Slot()));//
	connect(axes_mode_Action, SIGNAL(triggered()), this, SLOT(axes_mode_Slot()));
	connect(axes_display_edges_Action, SIGNAL(triggered()), this, SLOT(axes_display_edges_Slot()));
	connect(axes_new_Action, SIGNAL(triggered()), this, SLOT(axes_new_Slot()));

	connect(filter_reprocess_last_filter, SIGNAL(triggered()), this, SLOT(filter_reprocess_last_Slot()));

	connect(lines_display_unselected_listing_Action, SIGNAL(triggered()), this, SLOT(lines_display_unselected_listing_Slot()));
	connect(lines_display_unselected_GLview_Action, SIGNAL(triggered()), this, SLOT(lines_display_unselected_GLview_Slot()));
	connect(lines_display_zombies_listing_Action, SIGNAL(triggered()), this, SLOT(lines_display_zombies_listing_Slot()));
	connect(lines_display_zombies_GLview_Action, SIGNAL(triggered()), this, SLOT(lines_display_zombies_GLview_Slot()));
        
	connect(tools_new_format_Action, SIGNAL(triggered()), this, SLOT(new_format_Slot()));
	connect(tools_cur_format_Action, SIGNAL(triggered()), this, SLOT(cur_format_Slot()));

	//connect(whats_this_Action, SIGNAL(triggered()), this, SLOT(whats_this_Slot()));
	connect(about_Action, SIGNAL(triggered()), this, SLOT(about_Slot()));
}
