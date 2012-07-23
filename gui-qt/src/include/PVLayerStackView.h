/**
 * \file PVLayerStackView.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVLAYERSTACKVIEW_H
#define PVLAYERSTACKVIEW_H


#include <QTableView>
#include <QEvent>
#include <QPoint>

#include <PVLayerStackDelegate.h>
#include <PVLayerStackEventFilter.h>
#include <PVLayerStackModel.h>

namespace PVInspector {
class PVMainWindow;
class PVLayerStackWidget;

/**
 *  \class PVLayerStackView
 */
class PVLayerStackView : public QTableView
{
	Q_OBJECT

		PVMainWindow *main_window;

		PVLayerStackDelegate *layer_stack_delegate;
		PVLayerStackEventFilter *layer_stack_event_filter;

	public:
		int mouse_hover_layer_index; // FIXME, private
		int last_mouse_hover_layer_index; // FIXME, private
		PVLayerStackView(PVMainWindow *mw, PVLayerStackModel *model, PVLayerStackWidget *parent);

		void leaveEvent(QEvent *event);
//		void set_model(PVLayerStackModel *pv_listing_model);

		inline PVLayerStackWidget* get_parent() { return _parent; }

	protected:
		void save_layer(int idx);
		void import_layer();
		void save_layer_stack();
		void load_layer_stack();

	protected slots:
		void show_ctxt_menu(QPoint const& pt);

	protected:
		// Context menu
		QMenu* _ctxt_menu;
		QAction* _ctxt_menu_save_act;
		QAction* _ctxt_menu_load_act;
		QAction* _ctxt_menu_save_ls_act;
		QAction* _ctxt_menu_load_ls_act;
		QAction* _ctxt_menu_set_sel_layer;
		PVLayerStackWidget* _parent;
};
}

#endif // PVLAYERSTACKVIEW_H
