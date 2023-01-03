/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVMAINWINDOW_H
#define PVMAINWINDOW_H

#include <QMainWindow>

#include <QFile>
#include <QStackedWidget>

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVMeanValue.h>

#include <pvkernel/rush/PVInput.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVSourceCreatorFactory.h>
#include <pvkernel/widgets/PVFileDialog.h>

#include <inendi/PVLayerFilter.h>
#include <inendi/PVSelection.h>

#include <pvguiqt/PVProjectsTabWidget.h>
#include <pvguiqt/PVAboutBoxDialog.h>

#include <PVFilesTypesSelWidget.h>

#include <tbb/task_scheduler_init.h>

QT_BEGIN_NAMESPACE
class QAction;
class QMenu;
class QPlainTextEdit;
QT_END_NAMESPACE

namespace PVCore
{
class PVSerializeArchive;
} // namespace PVCore

namespace PVGuiQt
{
class PVSourceWorkspace;
class PVAboutBoxDialog;
class PVExportSelectionDlg;
} // namespace PVGuiQt

namespace PVInspector
{

class PVMainWindow;
class PVStartScreenWidget;

/**
 * \class PVMainWindow
 */
class PVMainWindow : public QMainWindow
{
	Q_OBJECT

	friend class PVStartScreenWidget;

  public:
	PVMainWindow(QWidget* parent = 0);

	PVGuiQt::PVProjectsTabWidget* _projects_tab_widget;

	QMenuBar* menubar;
	QMenu* filter_Menu;

	char* last_sendername;
	bool report_started;
	int report_image_index;
	QString* report_filename;

	Inendi::PVView* current_view() { return get_root().current_view(); }
	Inendi::PVView const* current_view() const { return get_root().current_view(); }

	Inendi::PVScene* current_scene() { return get_root().current_scene(); }
	Inendi::PVScene const* current_scene() const { return get_root().current_scene(); }

	void move_selection_to_new_layer(Inendi::PVView* view);
	void commit_selection_to_new_layer(Inendi::PVView* view);
	void set_color(Inendi::PVView* view);

	void import_type(PVRush::PVInputType_p in_t);
	void import_type(PVRush::PVInputType_p in_t,
	                 PVRush::PVInputType::list_inputs const& inputs,
	                 PVRush::hash_formats& formats,
	                 PVRush::hash_format_creator& format_creator,
	                 QString const& choosenFormat);
	void load_files(std::vector<QString> const& files, QString format);
	/* void import_type(); */

	QString get_solution_path() const { return get_root().get_path(); }

	void set_window_title_with_filename();

	bool maybe_save_solution();

  protected:
	void remove_source(Inendi::PVSource* src_p);

  protected:
	bool event(QEvent* event) override;

  public Q_SLOTS:
	void about_Slot(PVGuiQt::PVAboutBoxDialog::Tab tab);
	void commit_selection_to_new_layer_Slot();
	void move_selection_to_new_layer_Slot();
	void selection_set_from_current_layer_Slot();
	void selection_set_from_layer_Slot();
	void export_selection_Slot();

	/**
	 * Export selection and import it on mineset using there REST API.
	 */
	void export_selection_to_mineset_Slot();

	void filter_Slot();
	void new_format_Slot();
	void cur_format_Slot();
	void edit_format_Slot(const QString& format);
	void open_format_Slot();
	void filter_reprocess_last_Slot();
	void import_type_default_Slot();
	void import_type_Slot();
	void import_type_Slot(const QString& itype);
	void events_display_unselected_listing_Slot();
	void events_display_zombies_listing_Slot();
	void events_display_unselected_zombies_parallelview_Slot();
	bool load_source_from_description_Slot(PVRush::PVSourceDescription);
	Inendi::PVScene& project_new_Slot();
	void quit_Slot();
	void selection_all_Slot();
	void selection_inverse_Slot();
	void selection_none_Slot();
	void enable_menu_filter_Slot(bool);
	void set_color_Slot();
	void view_display_inv_elts_Slot();
	void get_screenshot_widget();
	QScreen* get_screen() const;
	void get_screenshot_window();
	void get_screenshot_desktop();
	// Called by input_type plugins to edit a format.
	// Not an elegant solution, must find better.
	void edit_format_Slot(QString const& path, QWidget* parent);
	void edit_format_Slot(QDomDocument& doc, QWidget* parent);
	void axes_combination_editor_Slot();

	void solution_new_Slot();
	void solution_load_Slot();
	void solution_save_Slot();
	void solution_saveas_Slot();

	void close_solution_Slot();

  protected:
	void closeEvent(QCloseEvent* event) override;

  private:
	void display_inv_elts();

	void save_screenshot(const QPixmap& pixmap, const QString& title, const QString& name);

  private Q_SLOTS:
	void root_modified();
	bool load_solution(QString const& file);
	void load_solution_and_create_mw(QString const& file);
	void set_auto_detect_cancellation(bool cancel = true) { _auto_detect_cancellation = cancel; }
	void menu_activate_is_file_opened(bool cond);

  private:
	void connect_actions();
	void create_actions();
	void create_menus();
	void create_filters_menu_and_actions();
	void create_actions_import_types(QMenu* menu);

  private:
	bool is_project_untitled() { return _projects_tab_widget->is_current_project_untitled(); }
	bool load_source(Inendi::PVSource* src, bool update_recent_items = true);
	void source_loaded(Inendi::PVSource& src, bool update_recent_items);
	void flag_investigation_as_cached(const QString& file);

  private:
	QMenu* file_Menu;
	QMenu* events_Menu;
	QMenu* selection_Menu;
	QMenu* tools_Menu;
	QMenu* source_Menu;
	QMenu* view_Menu;
	QMenu* help_Menu;

	QAction* about_Action;
	QAction* refman_Action;
	QAction* axes_combination_editor_Action;
	QAction* events_display_unselected_listing_Action;
	QAction* events_display_zombies_listing_Action;
	QAction* events_display_unselected_zombies_parallelview_Action;
	QAction* commit_selection_to_new_layer_Action;
	QAction* move_selection_to_new_layer_Action;
	QAction* filter_reprocess_last_filter;
	QAction* project_new_Action;
	QAction* solution_new_Action;
	QAction* solution_load_Action;
	QAction* solution_save_Action;
	QAction* solution_saveas_Action;
	QAction* export_selection_Action;
	QAction* export_selection_to_mineset_Action; //!< Menu to trigger mineset export
	QAction* new_file_Action;
	QAction* new_scene_Action;
	QAction* quit_Action;
	QAction* select_scene_Action;
	QAction* selection_all_Action;
	QAction* selection_inverse_Action;
	QAction* selection_none_Action;
	QAction* selection_from_current_layer_Action;
	QAction* selection_from_layer_Action;
	QAction* set_color_Action;
	QAction* tools_new_format_Action;
	QAction* tools_open_format_Action;
	QAction* tools_cur_format_Action;
	QAction* view_Action;
	QAction* view_display_inv_elts_Action;

	QSpacerItem* pv_mainSpacerTop;
	QSpacerItem* pv_mainSpacerBottom;
	QWidget* pv_centralMainWidget;
	QStackedWidget* pv_centralWidget;
	QVBoxLayout* pv_mainLayout;
	QVBoxLayout* pv_startLayout;
	PVWidgets::PVFileDialog _load_solution_dlg;

	QString _current_save_root_folder;

  protected:
	void keyPressEvent(QKeyEvent* event) override;
	void treat_invalid_formats(QHash<QString, std::pair<QString, QString>> const& errors);

  public:
	Inendi::PVRoot& get_root();
	Inendi::PVRoot const& get_root() const;

  private:
	static PVMainWindow* find_main_window(const QString& path);
	bool is_solution_untitled() const { return get_solution_path().isEmpty(); }
	void save_solution(QString const& file, bool save_log_file = false);
	void reset_root();
	void close_solution();

	std::string get_next_scene_name();

  Q_SIGNALS:
	void change_of_current_view_Signal();
	void filter_applied_Signal();
	void zombie_mode_changed_Signal();

  private:
	QString _cur_project_file;
	static int sequence_n;
	Inendi::PVRoot _root;
	bool _auto_detect_cancellation;

  private:
	QString _screenshot_root_dir;
};
} // namespace PVInspector

#endif // PVMAINWINDOW_H
