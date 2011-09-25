//! \file PVMainWindowMenusActions.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <PVMainWindow.h>
#include <PVInputTypeMenuEntries.h>

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

	// The new_file Action
	new_file_Action = new QAction(tr("&New"), this);
	new_file_Action->setIcon(QIcon(":/document-new.png"));
	new_file_Action->setShortcut(QKeySequence::New);
	new_file_Action->setStatusTip(tr("Create a new file."));
	new_file_Action->setWhatsThis(tr("Use this to create a new file."));

	// The importFile Action
	//	import_file_Action = new QAction(tr("&Import"), this);
	//	import_file_Action->setToolTip(tr("Import a file."));
	//	import_file_Action->setShortcut(QKeySequence::Italic);

	// Export our selection Action
	export_selection_Action = new QAction(tr("Export &selection..."), this);
	export_selection_Action->setToolTip(tr("Export our current selection"));

	// The extractorFile Action
	extractor_file_Action = new QAction(tr("&Extractor..."), this);
	extractor_file_Action->setToolTip(tr("Launch the Picviz Extractor"));

	file_format_builder_Action = new QAction(tr("Format Builder..."), this);


	//	remote_log_Action = new QAction(tr("Import a &remote file"), this);
	//	remote_log_Action->setToolTip(tr("Import a remote file."));
	//	remote_log_Action->setShortcut(Qt::ControlModifier + Qt::Key_R);

	// #ifdef CUSTOMER_RELEASE
	// 	// The openFile Action
	// 	open_file_Action = new QAction(tr("&Open"), this);
	// 	open_file_Action->setIcon(QIcon(":/document-open.png"));
	// 	open_file_Action->setShortcut(QKeySequence::Open);
	// 	open_file_Action->setStatusTip(tr("Open a file."));
	// 	open_file_Action->setToolTip(tr("Open a file."));
	// 	open_file_Action->setWhatsThis(tr("Use this to open a file."));

	// 	// The saveFile Action
	// 	save_file_Action = new QAction(tr("&Save"), this);
	// 	save_file_Action->setToolTip(tr("Export a file"));
	// 	save_file_Action->setShortcut(QKeySequence::Save);
	// #endif	// CUSTOMER_RELEASE

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

	//commit_selection_in_current_layer_Action = new QAction(tr("Keep &current layer"), this);
	//commit_selection_in_current_layer_Action->setShortcut(QKeySequence(Qt::Key_K));
	commit_selection_to_new_layer_Action = new QAction(tr("Create new layer from selection"), this);
	commit_selection_to_new_layer_Action->setShortcut(QKeySequence(Qt::ALT + Qt::Key_K));

	/************************
	 * For the "Scene" menu entry
	 ************************/
	new_scene_Action = new QAction(tr("&New Scene"), this);
	select_scene_Action = new QAction(tr("&Select Scene"), this);

	/************************
	 * For the "View" menu entry
	 ************************/
	view_open_Action = new QAction(tr("&Open"), this);
	view_save_Action = new QAction(tr("&Save"), this);
	view_show_new_Action = new QAction(tr("New Parallel view"), this);
	view_new_scatter_Action = new QAction(tr("New Scatter view"), this);

	/***************************
	 * For the "Axes" menu entry
	 ***************************/
	axes_editor_Action = new QAction(tr("Edit Axes"), this);
	axes_mode_Action = new QAction(tr("Enter Axes mode"), this);
	axes_mode_Action->setShortcut(QKeySequence(Qt::Key_X));
	axes_display_edges_Action = new QAction(tr("Display Edges"), this);
	axes_display_edges_Action->setShortcut(QKeySequence(Qt::Key_Y));

	/***************************
	 * For the "Lines" menu entry
	 ***************************/
	lines_display_unselected_Action = new QAction(tr("Hide unselected lines"), this);
	lines_display_unselected_Action->setShortcut(QKeySequence(Qt::Key_U));
	lines_display_unselected_listing_Action = new QAction(tr("Hide unselected lines in listing"), this);
	lines_display_unselected_listing_Action->setShortcut(QKeySequence(Qt::ALT + Qt::Key_U));
	lines_display_unselected_GLview_Action = new QAction(tr("Hide unselected lines in view"), this);
	lines_display_unselected_GLview_Action->setShortcut(QKeySequence(Qt::SHIFT + Qt::Key_U));
	lines_display_zombies_Action = new QAction(tr("Hide zombies lines"), this);
	lines_display_zombies_Action->setShortcut(QKeySequence(Qt::Key_Z));
	lines_display_zombies_listing_Action = new QAction(tr("Hide zombies lines in listing"), this);
	lines_display_zombies_listing_Action->setShortcut(QKeySequence(Qt::ALT + Qt::Key_Z));
	lines_display_zombies_GLview_Action = new QAction(tr("Hide zombies lines in view"), this);
	lines_display_zombies_GLview_Action->setShortcut(QKeySequence(Qt::SHIFT + Qt::Key_Z));

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
	//file_Menu->addAction(new_file_Action);
	create_actions_import_types(file_Menu);
	file_Menu->addSeparator();
	file_Menu->addAction(export_selection_Action);
	file_Menu->addAction(extractor_file_Action);
	file_Menu->addSeparator();
	file_Menu->addAction(file_format_builder_Action);
	file_Menu->addSeparator();

	// #ifdef CUSTOMER_RELEASE
	// 	file_Menu->addAction(open_file_Action);
	// 	file_Menu->addAction(save_file_Action);
	// #endif

	// 	file_Menu->addSeparator();
	//file_Menu->addAction(export_file_Action);
	//file_Menu->addSeparator();
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


	selection_Menu = menubar->addMenu(tr("Se&lection"));
	selection_Menu->addAction(selection_all_Action);
	selection_Menu->addAction(selection_none_Action);
	selection_Menu->addAction(selection_inverse_Action);
	selection_Menu->addSeparator();
	selection_Menu->addAction(set_color_Action);
	selection_Menu->addSeparator();
	//selection_Menu->addAction(commit_selection_in_current_layer_Action);
	selection_Menu->addAction(commit_selection_to_new_layer_Action);
	selection_Menu->addSeparator();

	filter_Menu = menubar->addMenu(tr("Fil&ters"));
	create_filters_menu_and_actions();

	// layer_Menu = menubar->addMenu(tr("&Layers"));
	// layer_Menu->addAction(commit_selection_in_current_layer_Action);
	// layer_Menu->addAction(commit_selection_to_new_layer_Action);

	//scene_Menu = menubar->addMenu(tr("S&cene"));
	//scene_Menu->addAction(new_scene_Action);
	//scene_Menu->addAction(select_scene_Action);

#ifdef CUSTOMER_RELEASE
	view_Menu = menubar->addMenu(tr("&View"));
	view_Menu->addAction(view_open_Action);
	view_Menu->addAction(view_save_Action);
	view_Menu->addSeparator();
	view_Menu->addAction(view_show_new_Action);
	view_Menu->addAction(view_new_scatter_Action);
#endif

	axes_Menu = menubar->addMenu(tr("Axes"));
	axes_Menu->addAction(axes_editor_Action);
	axes_Menu->addSeparator();
	axes_Menu->addAction(axes_mode_Action);
	axes_Menu->addAction(axes_display_edges_Action);

	lines_Menu = menubar->addMenu(tr("Lines"));
	lines_Menu->addAction(lines_display_unselected_Action);
	lines_Menu->addAction(lines_display_unselected_listing_Action);
	lines_Menu->addAction(lines_display_unselected_GLview_Action);
	lines_Menu->addSeparator();
	lines_Menu->addAction(lines_display_zombies_Action);
	lines_Menu->addAction(lines_display_zombies_listing_Action);
	lines_Menu->addAction(lines_display_zombies_GLview_Action);

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
	extractor_file_Action->setEnabled(cond);
	export_selection_Action->setEnabled(cond);

	axes_Menu->setEnabled(cond);
	filter_Menu->setEnabled(cond);
	lines_Menu->setEnabled(cond);
	selection_Menu->setEnabled(cond);
	view_Menu->setEnabled(cond);
	windows_Menu->setEnabled(cond);
}

