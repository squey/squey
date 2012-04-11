//! \file PVListingView.h
//! $Id: PVListingView.h 3092 2011-06-09 06:43:23Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVLISTINGVIEW_H
#define PVLISTINGVIEW_H

#include <QTableView>
#include <QHeaderView>

#include <picviz/PVView.h>

#include <PVListingModel.h>
#include <PVLayerStackModel.h>


namespace PVInspector {
class PVMainWindow;
class PVTabSplitter;
class PVLayerFilterProcessWidget;
class PVListingSortFilterProxyModel;

/**
 * \class PVListingView
 */
class PVListingView : public QTableView
{
	Q_OBJECT

	PVMainWindow    *main_window; //<!
	PVTabSplitter   *_parent;
	Picviz::PVView_p lib_view;    //<!

public slots:
	void slotDoubleClickOnVHead(int);

// FIXME!			void update_row_count_in_all_dynamic_listing_model_Slot();
			/* void update_to_current_selection_Slot();*/

/*			void refresh_listing_Slot();
			void refresh_listing_with_horizontal_header_Slot();
      void selection_changed_Slot();
			void update_pv_listing_model_Slot();*/

public:
// Removed (=> PVTabSplitter) PVLayerStackModel *pv_layer_stack_model; // FIXME: should be private

	/**
	 * Constructor.
	 *
	 * @param mw
	 * @param lib_view
	 * @param parent
	 */
	PVListingView(PVMainWindow *mw, PVTabSplitter *parent);

	/**
	 *
	 * @return
	 */
//	picviz_view_t *get_lib_view()const{return lib_view;}

	/**
	 *
	 * @return
	 */
//	int get_screenshot_index();

	/**
	 *
	 */
//	void increment_screenshot_index();

	void refresh_listing_filter();
	void keyEnterPressed();
	void wheelEvent(QWheelEvent* e);

	void update_view();

	PVListingSortFilterProxyModel* get_listing_model();

public slots:
	void selectAll();
	void corner_button_clicked();

protected:
	void mouseDoubleClickEvent(QMouseEvent* event);

private:
	QVector<PVRow> get_selected_rows();
	void selectionChanged(const QItemSelection &selected, const QItemSelection &deselected);

private:
	void process_ctxt_menu_action(QAction* act);
	void process_ctxt_menu_copy();
	void process_ctxt_menu_set_color();

private:
	void update_view_selection_from_listing_selection();

private slots:
	void show_ctxt_menu(const QPoint& pos);
	void show_hhead_ctxt_menu(const QPoint& pos);
	void set_color_selected(const QColor& color);

private:
	QMenu* _ctxt_menu;
	QMenu* _hhead_ctxt_menu;
	QAction* _action_col_sort;
	QAction* _action_col_unique;
	bool _show_ctxt_menu;
	PVRow _ctxt_row;
	PVCol _ctxt_col;
	QString _ctxt_v;
	PVCore::PVArgumentList _ctxt_args;
	PVLayerFilterProcessWidget* _ctxt_process;
	QAction* _act_copy;
	QAction* _act_set_color;
};

}

#endif // PVLISTINGVIEW_H
