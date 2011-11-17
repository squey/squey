//! \file PVLayerStackWidget.h
//! $Id: PVLayerStackWidget.h 2496 2011-04-25 14:10:00Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVLAYERSTACKWIDGET_H
#define PVLAYERSTACKWIDGET_H

#include <QAction>
#include <QWidget>

#include <PVLayerStackModel.h>
#include <PVLayerStackView.h>

namespace PVInspector {
class PVMainWindow;
class PVTabSplitter;

/**
 *  \class PVLayerStackWidget
 */
class PVLayerStackWidget : public QWidget
{
	Q_OBJECT

		PVMainWindow     *main_window;            //!<
		PVTabSplitter    *parent_tab;             //!<
		PVLayerStackView *pv_layer_stack_view;    //!<

		void create_actions(QToolBar *toolbar);
	public:
		/**
		 * Constructor.
		 */
		PVLayerStackWidget(PVMainWindow *mw, PVLayerStackModel *model, PVTabSplitter *parent);

		/**
		 *
		 * @return
		 */
		PVLayerStackView *get_layer_stack_view()const{return pv_layer_stack_view;}
		PVTabSplitter* get_parent_tab() { return parent_tab; }

	public slots:
		void delete_layer_Slot();
		void duplicate_layer_Slot();
		void move_down_Slot();
		void move_up_Slot();
		void new_layer_Slot();
//		void show_hide_layer_stack_Slot(bool visible);

	public:
		void refresh();
};
}

#endif // PVLAYERSTACKWIDGET_H


