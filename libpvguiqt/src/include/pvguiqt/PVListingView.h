/**
 * \file PVListingView.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVLISTINGVIEW_H
#define PVLISTINGVIEW_H

#include <QTableView>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>

#include <pvhive/PVActor.h>
#include <pvhive/PVObserverSignal.h>

#include <picviz/PVView_types.h>

namespace PVGuiQt {

//class PVLayerFilterProcessWidget;
class PVListingSortFilterProxyModel;

/**
 * \class PVListingView
 */
class PVListingView : public QTableView
{
	Q_OBJECT

public:
	PVListingView(Picviz::PVView_sp& view, QWidget* parent = NULL);

	void refresh_listing_filter();
	void wheelEvent(QWheelEvent* e);

	PVListingSortFilterProxyModel* get_listing_model();

public slots:
	void selectAll();
	void corner_button_clicked();

protected:
	void mouseDoubleClickEvent(QMouseEvent* event);
	void keyPressEvent(QKeyEvent* event) override;

private:
	QVector<PVRow> get_selected_rows();
	void selectionChanged(const QItemSelection &selected, const QItemSelection &deselected);

private:
	void process_ctxt_menu_action(QAction* act);
	void process_ctxt_menu_copy();
	void process_ctxt_menu_set_color();

private:
	void update_view_selection_from_listing_selection();
	Picviz::PVView const& lib_view() const { return *_obs.get_object(); }
	Picviz::PVView& lib_view() { return *_obs.get_object(); }

private slots:
	void slotDoubleClickOnVHead(int);
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
	//PVLayerFilterProcessWidget* _ctxt_process;
	QAction* _act_copy;
	QAction* _act_set_color;

private:
	// Observers
	PVHive::PVObserverSignal<Picviz::PVView> _obs;

	// Actor
	PVHive::PVActor<Picviz::PVView> _actor;
};

}

#endif // PVLISTINGVIEW_H
