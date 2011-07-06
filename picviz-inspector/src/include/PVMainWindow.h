//! \file PVMainWindow.h
//! $Id: PVMainWindow.h 3196 2011-06-23 16:24:50Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVMAINWINDOW_H
#define PVMAINWINDOW_H

#include <QMainWindow>

#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>

#include <picviz/init.h>
#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>
#include <picviz/PVSource.h>
#include <picviz/PVMapped.h>
#include <picviz/PVView.h>
#include <picviz/PVLayerFilter.h>
#include <picviz/PVSelection.h>
#include <pvfilter/PVArgument.h>

#include <pvgl/general.h>
#include <pvgl/PVCom.h>

#include <PVAxisPropertiesWidget.h>
#include <PVColorDialog.h>
#include <PVDualSlider.h>
#include <PVExportSelectionDialog.h>
#include <PVFilterWidget.h>
#include <PVImportFileDialog.h>
//#include <PVMapWidget.h>
#include <PVOpenFileDialog.h>
#include <PVSaveFileDialog.h>
#include <PVListingView.h>
#include <PVListingsTabWidget.h>
#include <PVFilterSearchWidget.h>

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

			QDialog *about_dialog;
public:
	PVMainWindow(QWidget *parent = 0);

// XXX		picviz_datatreerootitem_t *datatree;

	PVAxisPropertiesWidget *pv_AxisProperties;
	PVColorDialog *pv_ColorDialog;

	PVFilterWidget *pv_FilterWidget;

	PVExportSelectionDialog *pv_ExportSelectionDialog;
	PVImportFileDialog *pv_ImportFileDialog;

//		PVMapWidget *pv_MapWidget;
	PVOpenFileDialog    *pv_OpenFileDialog;
	/* LogViewerWidget     *pv_RemoteLog; */
	PVSaveFileDialog    *pv_SaveFileDialog;
	PVTabSplitter       *current_tab;
	PVListingsTabWidget *pv_ListingsTabWidget;

	PVFilterSearchWidget *pv_FilterSearchWidget;

	QMainWindow *RemoteLogDialog;

	QMenuBar *menubar;
	QMenu *filter_Menu;

	char *last_sendername;
	Picviz::PVLayerFilter *filter;

	//Picviz::PVSource *import_source;
	Picviz::PVRoot_p root;

	/* QGridLayout *filter_widgets_layout; */
	void commit_selection_in_current_layer(Picviz::PVView_p view);
	void commit_selection_to_new_layer(Picviz::PVView_p view);
	void set_color(Picviz::PVView_p view);
	PVGL::PVCom* get_pvcom();
	
	static void process_layer_filter(Picviz::PVLayerFilter* filter, Picviz::PVLayer* layer);

public slots:
	void about_Slot();
	void axes_editor_Slot();
	void axes_mode_Slot();
	void axes_display_edges_Slot();
	void change_of_current_view_Slot();
	void commit_selection_in_current_layer_Slot();
	void commit_selection_to_new_layer_Slot();
	/* void dualslider_value_changed_Slot(); */
	void export_file_Slot();
	void export_selection_Slot();
	void extractor_file_Slot();
	void filter_select_all_Slot();
	void filter_Slot();
	void file_format_builder_Slot();
	void import_type_Slot();
	void lines_display_unselected_Slot();
	void lines_display_unselected_listing_Slot();
	void lines_display_unselected_GLview_Slot();
	void lines_display_zombies_Slot();
	void lines_display_zombies_listing_Slot();
	void lines_display_zombies_GLview_Slot();
	void map_Slot();
	void new_file_Slot();
	void new_scene_Slot();
	/* void open_file_Slot(); */
	void quit_Slot();
	void refresh_current_view_Slot();
	/* void save_file_Slot(); */
	void remote_log_Slot();
	void select_scene_Slot();
	void selection_all_Slot();
	void selection_inverse_Slot();
	void selection_none_Slot();
	void enable_menu_filter_Slot(bool);
	void set_color_Slot();
	void textedit_text_changed_Slot();
	void view_open_Slot();
	void view_save_Slot();
	void view_show_new_Slot();
	void view_new_scatter_Slot();
	void check_messages();	/* SLOT? NOT SLOT? */
	void update_reply_finished_Slot(QNetworkReply *reply);
	void whats_this_Slot();

protected:
	void closeEvent(QCloseEvent* event);

private:
	void connect_actions();
	void connect_widgets();
	void create_actions();
	void create_menus();
	void create_filters_menu_and_actions();
	void create_actions_import_types(QMenu* menu);
	void menu_activate_is_file_opened(bool cond);

	QMenu *axes_Menu;
	QMenu *file_Menu;
	QMenu *edit_Menu;
	QMenu *layer_Menu;
	QMenu *lines_Menu;
	QMenu *scene_Menu;
	QMenu *selection_Menu;
	QMenu *view_Menu;
	QMenu *windows_Menu;
	QMenu *help_Menu;


	QAction *about_Action;
	QAction *axes_editor_Action;
	QAction *axes_mode_Action;
	QAction *axes_display_edges_Action;
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
	QAction *export_file_Action;
	QAction *export_selection_Action;
	QAction *extractor_file_Action;
	QAction *file_format_builder_Action;
	QAction *new_file_Action;
	QAction *new_scene_Action;
	/* QAction *open_file_Action; */
	QAction *paste_Action;
	QAction *quit_Action;
	QAction *redo_Action;
	QAction *remote_log_Action;
	/* QAction *save_file_Action; */
	QAction *select_scene_Action;
	QAction *selection_all_Action;
	QAction *selection_inverse_Action;
	QAction *selection_none_Action;
	QAction *set_color_Action;
	QAction *undo_Action;
	QAction *undo_history_Action;
	QAction *view_Action;
	QAction *view_open_Action;
	QAction *view_save_Action;
	QAction *view_show_new_Action;
	QAction *view_new_scatter_Action;
	QAction *whats_this_Action;

protected:
	bool eventFilter(QObject *watched_object, QEvent *event);
	void keyPressEvent(QKeyEvent *event);
	int update_check();

signals:
	void change_of_current_view_Signal();
	void color_changed_Signal();
	void filter_applied_Signal();
	void commit_to_new_layer_Signal();
	void selection_changed_Signal();
	void zombie_mode_changed_Signal();

	// Communication with PVGL.
private:
	PVGL::PVThread *pvgl_thread;
	PVGL::PVCom    *pvgl_com;
	QTimer     *timer;

	void create_pvgl_thread();
public:
	/**
	 *  @param view
	 *  @param refresh_states
	 */
	void update_pvglview(Picviz::PVView_p view, int refresh_states);
	void destroy_pvgl_views(Picviz::PVView_p view);

private:
	tbb::task_scheduler_init init_parallel;
};
}

#endif // PVMAINWINDOW_H
