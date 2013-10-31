/**
 * \file PVLayerStackView.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVLAYERSTACKVIEW_H
#define PVLAYERSTACKVIEW_H

#include <QFileDialog>
#include <QTableView>

#include <picviz/PVLayer_types.h>
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

public:
	PVLayerStackModel* ls_model();

private:
	void save_layer(int idx);
	void import_layer();
	void save_layer_stack();
	void load_layer_stack();
	void set_current_selection_from_layer(int model_idx);
	void reset_layer_colors(int layer_idx);

	Picviz::PVLayer& get_layer_from_idx(int model_idx);

private slots:
	void show_ctxt_menu(QPoint const& pt);
	void layer_clicked(QModelIndex const& idx);
	void layer_double_clicked(QModelIndex const& idx);

protected:
	void enterEvent(QEvent *event) override;
	void leaveEvent(QEvent *event) override;
	void mouseDoubleClickEvent(QMouseEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;


private:
	// Context menu
	QMenu* _ctxt_menu;
	QAction* _ctxt_menu_save_act;
	QAction* _ctxt_menu_load_act;
	QAction* _ctxt_menu_save_ls_act;
	QAction* _ctxt_menu_load_ls_act;
	QAction* _ctxt_menu_set_sel_layer;
	QAction* _ctxt_menu_reset_colors;

	QFileDialog _layer_dialog;
	QFileDialog _layerstack_dialog;
};

}

#endif // PVLAYERSTACKVIEW_H
