//! \file PVMainWindow.h
//! $Id: PVMainWindow.h 3196 2011-06-23 16:24:50Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVMAINWINDOW_H
#define PVMAINWINDOW_H

#include <QMainWindow>

#include <QFile>
#include <QLabel>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QSet>

#include <pvkernel/rush/PVInput.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVSourceCreatorFactory.h>

#include <picviz/init.h>
#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>
#include <picviz/PVSource.h>
#include <picviz/PVMapped.h>
#include <picviz/PVView.h>
#include <picviz/PVLayerFilter.h>
#include <picviz/PVSelection.h>
#include <pvkernel/core/PVArgument.h>

#include <pvsdk/PVMessenger.h>

#include <pvgl/general.h>
#include <pvgl/PVGLThread.h>

#include <PVAxisPropertiesWidget.h>
#include <PVDualSlider.h>
#include <PVExportSelectionDialog.h>
#include <PVFilterWidget.h>
//#include <PVMapWidget.h>
#include <PVOpenFileDialog.h>
#include <PVSaveFileDialog.h>
#include <PVListingsTabWidget.h>
#include <PVFilesTypesSelWidget.h>

//#include <>

/* #include <logviewer/logviewerwidget.h> */

#include <tbb/task_scheduler_init.h>


QT_BEGIN_NAMESPACE
class QAction;
class QMenu;
class QPlainTextEdit;
QT_END_NAMESPACE

namespace PVInspector {

/**
 * \class PVMainWindow
 */
class PVMainWindow : public QMainWindow
{
	Q_OBJECT

private:
	struct PVFormatDetectCtxt
	{
		PVFormatDetectCtxt(PVRush::PVInputType::list_inputs const& inputs_, QHash<QString,PVRush::input_type>& hash_input_name_, PVRush::hash_formats& formats_, PVRush::hash_format_creator& format_creator_, map_files_types& files_multi_formats_, QHash< QString,PVRush::PVInputType::list_inputs >& discovered_, QHash<QString, std::pair<QString,QString> >& formats_error_, PVRush::list_creators& lcr_, PVRush::PVInputType_p in_t_, QHash<QString,PVCore::PVMeanValue<float> >& discovered_types_):
			inputs(inputs_),
			hash_input_name(hash_input_name_),
			formats(formats_),
			format_creator(format_creator_),
			files_multi_formats(files_multi_formats_),
			discovered(discovered_),
			formats_error(formats_error_),
			lcr(lcr_),
			in_t(in_t_),
			discovered_types(discovered_types_)
		{ }

		PVRush::PVInputType::list_inputs const& inputs;
		QHash<QString,PVRush::input_type>& hash_input_name;
		PVRush::hash_formats& formats;
		PVRush::hash_format_creator& format_creator;
		map_files_types& files_multi_formats;
		QHash< QString,PVRush::PVInputType::list_inputs >& discovered;
		QHash<QString,std::pair<QString,QString> >& formats_error;
		PVRush::list_creators& lcr;
		PVRush::PVInputType_p in_t;
		QHash<QString,PVCore::PVMeanValue<float> >& discovered_types;
	};

private:
	QDialog *about_dialog;

public:
	PVMainWindow(QWidget *parent = 0);

// XXX		picviz_datatreerootitem_t *datatree;

	PVAxisPropertiesWidget *pv_AxisProperties;
	//PVColorDialog *pv_ColorDialog;

	PVFilterWidget *pv_FilterWidget;

	PVExportSelectionDialog *pv_ExportSelectionDialog;

//		PVMapWidget *pv_MapWidget;
	PVOpenFileDialog    *pv_OpenFileDialog;
	/* LogViewerWidget     *pv_RemoteLog; */
	PVSaveFileDialog    *pv_SaveFileDialog;
	PVTabSplitter       *current_tab;
	PVListingsTabWidget *pv_ListingsTabWidget;


	QMainWindow *RemoteLogDialog;

	QMenuBar *menubar;
	QMenu *filter_Menu;
	QLabel *statemachine_label;

	char *last_sendername;
	Picviz::PVLayerFilter *filter;


	bool report_started;
	int report_image_index;
	QString *report_filename;

	Picviz::PVRoot_p root;

	/* QGridLayout *filter_widgets_layout; */
	void commit_selection_in_current_layer(Picviz::PVView_p view);
	void commit_selection_to_new_layer(Picviz::PVView_p view);
	void refresh_view(Picviz::PVView_p view);
	void set_color(Picviz::PVView_p view);
	PVSDK::PVMessenger* get_pvmessenger();

	void import_type(PVRush::PVInputType_p in_t);
	void import_type(PVRush::PVInputType_p in_t, PVRush::PVInputType::list_inputs const& inputs, PVRush::hash_formats& formats, PVRush::hash_format_creator& format_creator, QString const& choosenFormat);
	void load_files(std::vector<QString> const& files, QString format);
	/* void import_type(); */
	void update_statemachine_label(Picviz::PVView_p view);

	void close_source(int index);
	void close_source(PVTabSplitter* tab);
	void close_scene();

	static QList<Picviz::PVView_p> list_displayed_picviz_views();

public slots:
	void about_Slot();
	void axes_editor_Slot();
	void axes_mode_Slot();
	void axes_display_edges_Slot();
	void axes_new_Slot();
	void change_of_current_view_Slot();
	void commit_selection_in_current_layer_Slot();
	void commit_selection_to_new_layer_Slot();
	void expand_selection_on_axis_Slot();
	void export_file_Slot();
	void export_selection_Slot();
	void extractor_file_Slot();
	void filter_select_all_Slot();
	void filter_Slot();
	void new_format_Slot();
	void cur_format_Slot();
	void import_type_default_Slot();
	void import_type_Slot();
	void lines_display_unselected_Slot();
	void lines_display_unselected_listing_Slot();
	void lines_display_unselected_GLview_Slot();
	void lines_display_zombies_Slot();
	void lines_display_zombies_listing_Slot();
	void lines_display_zombies_GLview_Slot();
	void map_Slot();
	void project_new_Slot();
	void project_load_Slot();
	bool project_save_Slot();
	bool project_saveas_Slot();
	void quit_Slot();
	void refresh_current_view_Slot();
	void select_scene_Slot();
	void selection_all_Slot();
	void selection_inverse_Slot();
	void selection_none_Slot();
	void enable_menu_filter_Slot(bool);
	void set_color_Slot();
	void textedit_text_changed_Slot();
	void view_new_parallel_Slot();
	void view_new_scatter_Slot();
	void view_screenshot_qt_Slot();
	void check_messages();	/* SLOT? NOT SLOT? */
	void update_reply_finished_Slot(QNetworkReply *reply);
	void whats_this_Slot();
	// Called by input_type plugins to edit a format.
	// Not an elegant solution, must find better.
	void edit_format_Slot(QString const& path, QWidget* parent);
	void edit_format_Slot(QDomDocument& doc, QWidget* parent);
	void set_color_selected(QColor const& color);
	void axes_combination_editor_Slot();

	void display_icon_Slot();

protected:
	void closeEvent(QCloseEvent* event);

private:
	bool load_project(const QString &file);
	bool save_project(const QString &file, PVCore::PVSerializeArchiveOptions_p options);
	void set_current_project_filename(const QString& file);
	bool maybe_save_project();
	bool is_project_untitled() { return _is_project_untitled; }
	void set_project_modified(bool modified);
	PVMainWindow* find_main_window(const QString& file);

private slots:
	void project_modified_Slot();
	void cur_format_changed_Slot();

private:
	void connect_actions();
	void connect_widgets();
	void create_actions();
	void create_menus();
	void create_filters_menu_and_actions();
	void create_actions_import_types(QMenu* menu);
	void menu_activate_is_file_opened(bool cond);

	// AG: that needs to be redesigned. I outlined this code as an automatic outliner would do, so that
	// a progress box can cancel this process.
	void auto_detect_formats(PVFormatDetectCtxt ctxt);

private:
	bool load_scene();
	bool load_source(Picviz::PVSource_p src);

	QMenu *axes_Menu;
	QMenu *file_Menu;
	QMenu *edit_Menu;
	QMenu *layer_Menu;
	QMenu *lines_Menu;
	QMenu *scene_Menu;
	QMenu *selection_Menu;
	QMenu* tools_Menu;
	QMenu *view_Menu;
	QMenu *windows_Menu;
	QMenu *help_Menu;


	QAction *about_Action;
	QAction *axes_editor_Action;
	QAction *axes_combination_editor_Action;
	QAction *axes_mode_Action;
	QAction *axes_display_edges_Action;
	QAction *axes_new_Action;
	QAction *expand_selection_on_axis_Action;
	QAction *lines_display_unselected_Action;
	QAction *lines_display_unselected_listing_Action;
	QAction *lines_display_unselected_GLview_Action;
	QAction *lines_display_zombies_Action;
	QAction *lines_display_zombies_listing_Action;
	QAction *lines_display_zombies_GLview_Action;
	QAction *copy_Action;
	QAction *commit_selection_in_current_layer_Action;
	QAction *commit_selection_to_new_layer_Action;
	QAction *cut_Action;
	QAction *project_new_Action;
	QAction *project_load_Action;
	QAction *project_save_Action;
	QAction *project_saveas_Action;
	QAction *export_file_Action;
	QAction *export_selection_Action;
	QAction *extractor_file_Action;
	QAction *new_file_Action;
	QAction *new_scene_Action;
	QAction *paste_Action;
	QAction *quit_Action;
	QAction *redo_Action;
	QAction *select_scene_Action;
	QAction *selection_all_Action;
	QAction *selection_inverse_Action;
	QAction *selection_none_Action;
	QAction *set_color_Action;
	QAction* tools_new_format_Action;
	QAction* tools_cur_format_Action;
	QAction *undo_Action;
	QAction *undo_history_Action;
	QAction *view_Action;
	QAction *view_new_parallel_Action;
	QAction *view_new_scatter_Action;
	QAction *view_screenshot_qt;
	QAction *whats_this_Action;
	
	QSpacerItem* pv_mainSpacerTop;
	QSpacerItem* pv_mainSpacerBottom;
	QWidget *pv_centralStartWidget;
	QWidget *pv_centralMainWidget;
	QStackedWidget* pv_centralWidget;
	QVBoxLayout *pv_mainLayout;
	QVBoxLayout *pv_startLayout;
	QLabel *pv_labelWelcomeIcon;
	QPixmap  *pv_welcomeIcon;
	QLabel* pv_lastCurVersion; 
	QLabel* pv_lastMajVersion; 

	QPushButton *pv_ImportFileButton;

protected:
	bool eventFilter(QObject *watched_object, QEvent *event);
	void keyPressEvent(QKeyEvent *event);
	int update_check();
	void treat_invalid_formats(QHash<QString, std::pair<QString,QString> > const& errors);
	PVTabSplitter* get_tab_from_view(Picviz::PVView_p picviz_view);
	void show_start_page(bool visible);
	void set_version_informations();

signals:
	void change_of_current_view_Signal();
	void color_changed_Signal();
	void filter_applied_Signal();
	void commit_to_new_layer_Signal();
	void selection_changed_Signal();
	void zombie_mode_changed_Signal();

	// Communication with PVGL.
private:
	PVGL::PVGLThread *pvgl_thread;
	PVSDK::PVMessenger  *pvsdk_messenger;
	QTimer     *timer;

	void create_pvgl_thread();
public:
	/**
	 *  @param view
	 *  @param refresh_states
	 */
	void update_pvglview(Picviz::PVView_p view, int refresh_states);
	void ensure_glview_exists(Picviz::PVView_p view);
	void destroy_pvgl_views(Picviz::PVView_p view);

private:
	tbb::task_scheduler_init init_parallel;

private:
	Picviz::PVScene_p _scene;
	QString _cur_project_file;
	bool _cur_project_save_everything;
	bool _is_project_untitled;

private:
	version_t _last_known_cur_release;
	version_t _last_known_maj_release;

};
}

#endif // PVMAINWINDOW_H
