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
#include <pvkernel/widgets/PVModdedIcon.h>

#include <QAction>
#include <QActionGroup>
#include <QDesktopServices>
#include <QMenuBar>

/******************************************************************************
 *
 * App::PVMainWindow::create_actions
 *
 *****************************************************************************/
void App::PVMainWindow::create_actions()
{
	PVLOG_DEBUG("App::PVMainWindow::%s\n", __FUNCTION__);
	/************************
	 * For the "File" menu entry
	 ************************/

	// The solution actions
	solution_new_Action = new QAction(tr("&New"), this);
	solution_new_Action->setShortcut(QKeySequence::New);
	solution_new_Action->setIcon(PVModdedIcon("folder-plus"));
	solution_load_Action = new QAction(tr("&Open"), this);
	solution_load_Action->setShortcut(QKeySequence::Open);
	solution_load_Action->setIcon(PVModdedIcon("folder-open"));
	solution_save_Action = new QAction(tr("&Save"), this);
	solution_save_Action->setShortcut(QKeySequence::Save);
	solution_save_Action->setIcon(PVModdedIcon("floppy-disk"));
	solution_saveas_Action = new QAction(tr("Save &as..."), this);
	solution_saveas_Action->setShortcut(QKeySequence::SaveAs);
	solution_saveas_Action->setIcon(PVModdedIcon("floppy-disk-circle-arrow-right"));

	// The new_file Action
	new_file_Action = new QAction(tr("&New"), this);
	new_file_Action->setIcon(QIcon(":/document-new.png"));
	new_file_Action->setShortcut(QKeySequence::New);
	new_file_Action->setStatusTip(tr("Create a new file."));
	new_file_Action->setWhatsThis(tr("Use this to create a new file."));

	// Export our selection Action
	export_selection_Action = new QAction(tr("E&xport"), this);
	export_selection_Action->setIcon(PVModdedIcon("file-export"));
	export_selection_Action->setToolTip(tr("Export the current selection"));

	quit_Action = new QAction(tr("&Quit"), this);
	quit_Action->setShortcut(QKeySequence::Quit);
	quit_Action->setIcon(PVModdedIcon("power-off"));

	/************************
	 * For the "Selection" menu entry
	 ************************/
	selection_all_Action = new QAction(tr("Select &all events"), this);
	selection_all_Action->setShortcut(QKeySequence(Qt::Key_A));
	selection_all_Action->setIcon(PVModdedIcon("square-full"));
	selection_none_Action = new QAction(tr("&Empty selection"), this);
	selection_none_Action->setIcon(PVModdedIcon("square"));
	selection_inverse_Action = new QAction(tr("&Invert selection"), this);
	selection_inverse_Action->setShortcut(QKeySequence(Qt::Key_I));
	selection_inverse_Action->setIcon(PVModdedIcon("square-half"));
	set_color_Action = new QAction(tr("Set color"), this);
	set_color_Action->setIcon(PVModdedIcon("palette"));
	set_color_Action->setShortcut(QKeySequence(Qt::Key_C));
	selection_from_current_layer_Action = new QAction(tr("Set selection from current layer"), this);
	selection_from_current_layer_Action->setIcon(PVModdedIcon("radio-checked"));
	selection_from_layer_Action = new QAction(tr("Set selection from layer..."), this);
	selection_from_layer_Action->setIcon(PVModdedIcon("selection-from-layer"));

	commit_selection_to_new_layer_Action = new QAction(tr("Create new layer from selection"), this);
	commit_selection_to_new_layer_Action->setShortcut(QKeySequence(Qt::ALT | Qt::Key_K));
	commit_selection_to_new_layer_Action->setIcon(PVModdedIcon("layer-from-selection"));
	move_selection_to_new_layer_Action = new QAction(tr("Move selection to new layer"), this);
	move_selection_to_new_layer_Action->setShortcut(QKeySequence(Qt::ALT | Qt::Key_M));
	move_selection_to_new_layer_Action->setIcon(PVModdedIcon("move-layer-from-selection"));

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

	/**************************
	 * For the "Settings" menu entry
	 **************************/
	auto* act_group_theme = new QActionGroup(this);
	const QString& settings_color_scheme = PVCore::PVTheme::settings_color_scheme();
	act_group_theme->setExclusive(true);
	settings_dark_theme_Action = new QAction(tr("&Dark"), act_group_theme);
	settings_dark_theme_Action->setCheckable(true);
	settings_dark_theme_Action->setChecked(settings_color_scheme == "dark");
	connect(settings_dark_theme_Action, &QAction::triggered, [](){
		PVCore::PVTheme::follow_system_scheme(false);
		PVCore::PVTheme::set_color_scheme(PVCore::PVTheme::EColorScheme::DARK);
	});
	//settings_dark_theme_Action->setIcon(PVModdedIcon("moon"));

	settings_light_theme_Action = new QAction(tr("&Light"), act_group_theme);
	settings_light_theme_Action->setCheckable(true);
	settings_light_theme_Action->setChecked(settings_color_scheme == "light");
	connect(settings_light_theme_Action, &QAction::triggered, [](){
		PVCore::PVTheme::follow_system_scheme(false);
		PVCore::PVTheme::set_color_scheme(PVCore::PVTheme::EColorScheme::LIGHT);
	});
	//settings_light_theme_Action->setIcon(PVModdedIcon("sun-bright"));

	settings_follow_system_theme_Action = new QAction(tr("&Auto"), act_group_theme);
	settings_follow_system_theme_Action->setCheckable(true);
	settings_follow_system_theme_Action->setChecked(settings_color_scheme == "system");
	connect(settings_follow_system_theme_Action, &QAction::triggered, [](){
		PVCore::PVTheme::follow_system_scheme(true);
		PVCore::PVTheme::set_color_scheme(PVCore::PVTheme::system_color_scheme());
	});
	//settings_follow_system_theme_Action->setIcon(PVModdedIcon("moon-over-sun"));

	/**************************
	 * For the "Help" menu entry
	 **************************/
	about_Action = new QAction(tr("&About"), this);
	about_Action->setIcon(PVModdedIcon("circle-info"));
	refman_Action = new QAction(tr("Reference &Manual"), this);
	refman_Action->setIcon(PVModdedIcon("book-open"));
}

/******************************************************************************
 *
 * App::PVMainWindow::create_menus
 *
 *****************************************************************************/
void App::PVMainWindow::create_menus()
{
	PVLOG_DEBUG("App::PVMainWindow::%s\n", __FUNCTION__);

	menubar = menuBar();
	
	file_Menu = menubar->addMenu(tr("&File"));
	file_Menu->setAttribute(Qt::WA_TranslucentBackground);
	auto* solution_Menu = new QMenu(tr("&Investigation"));
	solution_Menu->setIcon(PVModdedIcon("magnifying-glass-waveform"));
	solution_Menu->setAttribute(Qt::WA_TranslucentBackground);
	solution_Menu->addAction(solution_new_Action);
	solution_Menu->addAction(solution_load_Action);
	solution_Menu->addAction(solution_save_Action);
	solution_Menu->addAction(solution_saveas_Action);

	file_Menu->addMenu(solution_Menu);
	file_Menu->addSeparator();
	auto* import_Menu = new QMenu(tr("I&mport"));
	import_Menu->setIcon(PVModdedIcon("file-import"));
	import_Menu->setAttribute(Qt::WA_TranslucentBackground);
	create_actions_import_types(import_Menu);
	file_Menu->addMenu(import_Menu);
	file_Menu->addAction(export_selection_Action);
	file_Menu->addSeparator();
	file_Menu->addAction(quit_Action);

	selection_Menu = menubar->addMenu(tr("S&election"));
	selection_Menu->setAttribute(Qt::WA_TranslucentBackground);
	selection_Menu->addAction(selection_all_Action);
	selection_Menu->addAction(selection_none_Action);
	selection_Menu->addAction(selection_inverse_Action);
	selection_Menu->addSeparator();
	selection_Menu->addAction(selection_from_current_layer_Action);
	selection_Menu->addAction(selection_from_layer_Action);
	selection_Menu->addAction(commit_selection_to_new_layer_Action);
	selection_Menu->addAction(move_selection_to_new_layer_Action);
	selection_Menu->addSeparator();
	selection_Menu->addAction(set_color_Action);

	settings_Menu = menubar->addMenu(tr("&Settings"));
	settings_Menu->setAttribute(Qt::WA_TranslucentBackground);
	auto* theme_Menu = settings_Menu->addMenu(tr("&Theme"));
	theme_Menu->setIcon(PVModdedIcon("sun-bright"));
	theme_Menu->setAttribute(Qt::WA_TranslucentBackground);

	theme_Menu->addAction(settings_dark_theme_Action);
	theme_Menu->addAction(settings_light_theme_Action);
#ifdef __linux__
	if (QString(std::getenv("DISABLE_FOLLOW_SYSTEM_THEME")).isEmpty()) {
		theme_Menu->addAction(settings_follow_system_theme_Action);
	}
#endif

	help_Menu = menubar->addMenu(tr("&Help"));
	help_Menu->setAttribute(Qt::WA_TranslucentBackground);
	help_Menu->addAction(refman_Action);
	help_Menu->addAction(about_Action);
}

/******************************************************************************
 *
 * App::PVMainWindow::create_actions_import_types
 *
 *****************************************************************************/
void App::PVMainWindow::create_actions_import_types(QMenu* menu)
{
	PVGuiQt::PVInputTypeMenuEntries::add_inputs_to_menu(menu, this, SLOT(import_type_Slot()));
}

/******************************************************************************
 *
 * App::PVMainWindow::menu_activate_is_file_opened
 *
 *****************************************************************************/
void App::PVMainWindow::menu_activate_is_file_opened(bool cond)
{
	export_selection_Action->setEnabled(cond);

	selection_Menu->setEnabled(cond);
	tools_cur_format_Action->setEnabled(cond && is_solution_untitled());
	solution_save_Action->setEnabled(cond);
	solution_saveas_Action->setEnabled(cond);
}

/******************************************************************************
 *
 * App::PVMainWindow::connect_actions()
 *
 *****************************************************************************/
void App::PVMainWindow::connect_actions()
{
	PVLOG_DEBUG("App::PVMainWindow::%s\n", __FUNCTION__);
	connect(solution_new_Action, &QAction::triggered, this, &PVMainWindow::solution_new_Slot);
	connect(solution_load_Action, &QAction::triggered, this, &PVMainWindow::solution_load_Slot);
	connect(solution_save_Action, &QAction::triggered, this, &PVMainWindow::solution_save_Slot);
	connect(solution_saveas_Action, &QAction::triggered, this, &PVMainWindow::solution_saveas_Slot);

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

	connect(tools_new_format_Action, &QAction::triggered, this, &PVMainWindow::new_format_Slot);
	connect(tools_open_format_Action, &QAction::triggered, this, &PVMainWindow::open_format_Slot);
	connect(tools_cur_format_Action, &QAction::triggered, this, &PVMainWindow::cur_format_Slot);

	connect(about_Action, &QAction::triggered,
	        [this]() { about_Slot(PVGuiQt::PVAboutBoxDialog::Tab::SOFTWARE); });
	connect(refman_Action, &QAction::triggered,
	        []() { QDesktopServices::openUrl(QUrl(DOC_URL)); });
}
