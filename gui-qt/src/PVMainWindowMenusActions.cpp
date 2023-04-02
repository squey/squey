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

#include <PVMainWindow.h>
#include <pvguiqt/PVInputTypeMenuEntries.h>

#include <QAction>
#include <QDesktopServices>
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

	// The solution actions
	solution_new_Action = new QAction(tr("&New investigation"), this);
	solution_load_Action = new QAction(tr("&Load an investigation..."), this);
	solution_save_Action = new QAction(tr("&Save investigation"), this);
	solution_save_Action->setShortcut(QKeySequence::Save);
	solution_saveas_Action = new QAction(tr("S&ave investigation as..."), this);

	// The project actions
	project_new_Action = new QAction(tr("&New data collection"), this);

	// The new_file Action
	new_file_Action = new QAction(tr("&New"), this);
	new_file_Action->setIcon(QIcon(":/document-new.png"));
	new_file_Action->setShortcut(QKeySequence::New);
	new_file_Action->setStatusTip(tr("Create a new file."));
	new_file_Action->setWhatsThis(tr("Use this to create a new file."));

	// Export our selection Action
	export_selection_Action = new QAction(tr("&Selection..."), this);
	export_selection_Action->setToolTip(tr("Export the current selection"));

	quit_Action = new QAction(tr("&Quit"), this);
	quit_Action->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_Q));

	/************************
	 * For the "Selection" menu entry
	 ************************/
	selection_all_Action = new QAction(tr("Select &all events"), this);
	selection_all_Action->setShortcut(QKeySequence(Qt::Key_A));
	selection_none_Action = new QAction(tr("&Empty selection"), this);
	selection_inverse_Action = new QAction(tr("&Invert selection"), this);
	selection_inverse_Action->setShortcut(QKeySequence(Qt::Key_I));
	set_color_Action = new QAction(tr("Set color"), this);
	set_color_Action->setShortcut(QKeySequence(Qt::Key_C));
	selection_from_current_layer_Action = new QAction(tr("Set selection from current layer"), this);
	selection_from_layer_Action = new QAction(tr("Set selection from layer..."), this);

	commit_selection_to_new_layer_Action = new QAction(tr("Create new layer from selection"), this);
	commit_selection_to_new_layer_Action->setShortcut(QKeySequence(Qt::ALT | Qt::Key_K));
	move_selection_to_new_layer_Action = new QAction(tr("Move selection to new layer"), this);
	move_selection_to_new_layer_Action->setShortcut(QKeySequence(Qt::ALT | Qt::Key_M));

	/******************************
	 * For the "Filter" menu entry
	 ******************************/
	filter_reprocess_last_filter = new QAction(tr("Apply last filter..."), this);
	filter_reprocess_last_filter->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_F));

	/************************
	 * For the "Tools" menu entry
	 ************************/
	tools_new_format_Action = new QAction(tr("&New format..."), this);
	tools_open_format_Action = new QAction(tr("&Open format..."), this);
	tools_cur_format_Action = new QAction(tr("&Edit current format..."), this);

	/************************
	 * For the "Source" menu entry
	 ************************/
	view_display_inv_elts_Action = new QAction(tr("&Display invalid events..."), this);

	/***************************
	 * For the "View" menu entry
	 ***************************/
	axes_combination_editor_Action = new QAction(tr("Edit axes combination..."), this);

	/***************************
	 * For the "Events" menu entry
	 ***************************/
	events_display_unselected_listing_Action =
	    new QAction(tr("Toggle unselected events in listing"), this);
	events_display_unselected_listing_Action->setShortcut(QKeySequence(Qt::SHIFT | Qt::Key_U));

	events_display_zombies_listing_Action =
	    new QAction(tr("Toggle zombies events in listing"), this);
	events_display_zombies_listing_Action->setShortcut(QKeySequence(Qt::SHIFT | Qt::Key_Z));

	events_display_unselected_zombies_parallelview_Action =
	    new QAction(tr("Toggle unselected and zombies events"), this);
	events_display_unselected_zombies_parallelview_Action->setShortcut(QKeySequence(Qt::Key_U));

	/**************************
	 * For the "Help" menu entry
	 **************************/
	about_Action = new QAction(tr("&About"), this);
	refman_Action = new QAction(tr("Reference &Manual"), this);
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
	auto* solution_Menu = new QMenu(tr("&Investigation"));
	solution_Menu->addAction(solution_new_Action);
	solution_Menu->addAction(solution_load_Action);
	solution_Menu->addAction(solution_save_Action);
	solution_Menu->addAction(solution_saveas_Action);

	auto* project_Menu = new QMenu(tr("&Data collection"));
	project_Menu->addAction(project_new_Action);

	file_Menu->addMenu(solution_Menu);
	file_Menu->addSeparator();
	file_Menu->addMenu(project_Menu);
	file_Menu->addSeparator();
	file_Menu->addSeparator();
	file_Menu->addSeparator();
	auto* import_Menu = new QMenu(tr("I&mport"));
	create_actions_import_types(import_Menu);
	file_Menu->addMenu(import_Menu);
	auto* export_Menu = new QMenu(tr("E&xport"));
	export_Menu->addAction(export_selection_Action);
	file_Menu->addMenu(export_Menu);
	file_Menu->addSeparator();
	file_Menu->addAction(quit_Action);

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
	selection_Menu->addAction(commit_selection_to_new_layer_Action);
	selection_Menu->addAction(move_selection_to_new_layer_Action);
	selection_Menu->addSeparator();

	filter_Menu = menubar->addMenu(tr("Fil&ters"));
	filter_Menu->addAction(filter_reprocess_last_filter);
	filter_Menu->addSeparator();
	create_filters_menu_and_actions();

	source_Menu = menubar->addMenu(tr("&Source"));
	source_Menu->addAction(view_display_inv_elts_Action);

	view_Menu = menubar->addMenu(tr("&View"));
	view_Menu->addAction(axes_combination_editor_Action);

	events_Menu = menubar->addMenu(tr("&Events"));
	events_Menu->addAction(events_display_unselected_listing_Action);
	events_Menu->addAction(events_display_zombies_listing_Action);
	events_Menu->addSeparator();
	events_Menu->addAction(events_display_unselected_zombies_parallelview_Action);

	tools_Menu = menubar->addMenu(tr("F&ormat"));
	tools_Menu->addAction(tools_new_format_Action);
	tools_Menu->addAction(tools_open_format_Action);
	tools_Menu->addAction(tools_cur_format_Action);

	help_Menu = menubar->addMenu(tr("&Help"));
	help_Menu->addAction(about_Action);
	help_Menu->addAction(refman_Action);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::create_actions_import_types
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::create_actions_import_types(QMenu* menu)
{
	PVGuiQt::PVInputTypeMenuEntries::add_inputs_to_menu(menu, this, SLOT(import_type_Slot()));
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::menu_activate_is_file_opened
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::menu_activate_is_file_opened(bool cond)
{
	export_selection_Action->setEnabled(cond);

	filter_Menu->setEnabled(cond);
	events_Menu->setEnabled(cond);
	selection_Menu->setEnabled(cond);
	tools_cur_format_Action->setEnabled(cond && is_solution_untitled());
	source_Menu->setEnabled(cond);
	view_Menu->setEnabled(cond);
	solution_save_Action->setEnabled(cond);
	solution_saveas_Action->setEnabled(cond);
}

/******************************************************************************
 *
 * PVInspector::PVMainWindow::connect_actions()
 *
 *****************************************************************************/
void PVInspector::PVMainWindow::connect_actions()
{
	PVLOG_DEBUG("PVInspector::PVMainWindow::%s\n", __FUNCTION__);
	connect(solution_new_Action, &QAction::triggered, this, &PVMainWindow::solution_new_Slot);
	connect(solution_load_Action, &QAction::triggered, this, &PVMainWindow::solution_load_Slot);
	connect(solution_save_Action, &QAction::triggered, this, &PVMainWindow::solution_save_Slot);
	connect(solution_saveas_Action, &QAction::triggered, this, &PVMainWindow::solution_saveas_Slot);

	connect(project_new_Action, SIGNAL(triggered()), this,
	        SLOT(project_new_Slot())); // new connect syntax breaks compilation
	connect(export_selection_Action, &QAction::triggered, this,
	        &PVMainWindow::export_selection_Slot);
	connect(quit_Action, &QAction::triggered, this, &PVMainWindow::quit_Slot);

	connect(view_display_inv_elts_Action, &QAction::triggered, this,
	        &PVMainWindow::view_display_inv_elts_Slot);

	connect(selection_all_Action, &QAction::triggered, this, &PVMainWindow::selection_all_Slot);
	connect(selection_none_Action, &QAction::triggered, this, &PVMainWindow::selection_none_Slot);
	connect(selection_inverse_Action, &QAction::triggered, this,
	        &PVMainWindow::selection_inverse_Slot);
	connect(selection_from_current_layer_Action, &QAction::triggered, this,
	        &PVMainWindow::selection_set_from_current_layer_Slot);
	connect(selection_from_layer_Action, &QAction::triggered, this,
	        &PVMainWindow::selection_set_from_layer_Slot);

	connect(set_color_Action, &QAction::triggered, this, &PVMainWindow::set_color_Slot);

	connect(commit_selection_to_new_layer_Action, &QAction::triggered, this,
	        &PVMainWindow::commit_selection_to_new_layer_Slot);
	connect(move_selection_to_new_layer_Action, &QAction::triggered, this,
	        &PVMainWindow::move_selection_to_new_layer_Slot);

	connect(axes_combination_editor_Action, &QAction::triggered, this,
	        &PVMainWindow::axes_combination_editor_Slot); //

	connect(filter_reprocess_last_filter, &QAction::triggered, this,
	        &PVMainWindow::filter_reprocess_last_Slot);

	connect(events_display_unselected_listing_Action, &QAction::triggered, this,
	        &PVMainWindow::events_display_unselected_listing_Slot);
	connect(events_display_zombies_listing_Action, &QAction::triggered, this,
	        &PVMainWindow::events_display_zombies_listing_Slot);
	connect(events_display_unselected_zombies_parallelview_Action, &QAction::triggered, this,
	        &PVMainWindow::events_display_unselected_zombies_parallelview_Slot);

	connect(tools_new_format_Action, &QAction::triggered, this, &PVMainWindow::new_format_Slot);
	connect(tools_open_format_Action, &QAction::triggered, this, &PVMainWindow::open_format_Slot);
	connect(tools_cur_format_Action, &QAction::triggered, this, &PVMainWindow::cur_format_Slot);

	connect(about_Action, &QAction::triggered,
	        [this]() { about_Slot(PVGuiQt::PVAboutBoxDialog::Tab::SOFTWARE); });
	connect(refman_Action, &QAction::triggered,
	        []() { QDesktopServices::openUrl(QUrl(DOC_URL)); });
}
