/**
 * \file PVLayerStackView.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVLAYERSTACKVIEW_H
#define PVLAYERSTACKVIEW_H

#include <QFileDialog>
#include <QTableView>

#include <picviz/PVView_types.h>

namespace PVGuiQt {

class PVLayerStackModel;

/**
 *  \class PVLayerStackView
 */
class PVLayerStackView : public QTableView
{
	Q_OBJECT

public:
	PVLayerStackView(QWidget* parent = NULL);

	void leaveEvent(QEvent *event);

private:
	void save_layer(int idx);
	void import_layer();
	void save_layer_stack();
	void load_layer_stack();

private slots:
	void show_ctxt_menu(QPoint const& pt);
	void layer_clicked(QModelIndex const& idx);

public:
	PVLayerStackModel* ls_model();


private:
	// Context menu
	QMenu* _ctxt_menu;
	QAction* _ctxt_menu_save_act;
	QAction* _ctxt_menu_load_act;
	QAction* _ctxt_menu_save_ls_act;
	QAction* _ctxt_menu_load_ls_act;
	QAction* _ctxt_menu_set_sel_layer;

	QFileDialog _layer_dialog;
	QFileDialog _layerstack_dialog;
};

}

#endif // PVLAYERSTACKVIEW_H
