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

#include <QHeaderView>

namespace PVWidgets
{

class PVHelpWidget;

}

namespace PVGuiQt {

class PVLayerFilterProcessWidget;
class PVListingSortFilterProxyModel;

namespace __impl
{
class PVHorizontalHeaderView;
}

/**
 * \class PVListingView
 */
class PVListingView : public QTableView
{
	Q_OBJECT
	friend class PVStatsListingWidget;
	friend class PVHorizontalHeaderView;

public:
	PVListingView(Picviz::PVView_sp& view, QWidget* parent = NULL);

	void refresh_listing_filter();

	PVListingSortFilterProxyModel* get_listing_model();

public slots:
	void selectAll();
	void corner_button_clicked();
	void section_clicked(int col);

protected:
	void mouseDoubleClickEvent(QMouseEvent* event);
	void keyPressEvent(QKeyEvent* event) override;
	void wheelEvent(QWheelEvent* e);
	void reset() override;
	void resizeEvent(QResizeEvent * event) override;
	void enterEvent(QEvent* event) override;
	void leaveEvent(QEvent* event) override;
	void paintEvent(QPaintEvent * event) override;

signals:
	void resized();

private:
	QVector<PVRow> get_selected_rows();
	void extract_selection(Picviz::PVSelection &sel);
	void selectionChanged(const QItemSelection &selected, const QItemSelection &deselected);

private:
	void process_ctxt_menu_action(QAction* act);
	void process_ctxt_menu_copy();
	void process_ctxt_menu_set_color();

private:
	void update_view_selection_from_listing_selection();
	Picviz::PVView const& lib_view() const { return *_obs.get_object(); }
	Picviz::PVView& lib_view() { return *_obs.get_object(); }
	PVWidgets::PVHelpWidget* help_widget() { return _help_widget; }

private slots:
	void slotDoubleClickOnVHead(int);
	void show_ctxt_menu(const QPoint& pos);
	void show_hhead_ctxt_menu(const QPoint& pos);
	void set_color_selected(const PVCore::PVHSVColor& color);
	void columnResized(int column, int oldWidth, int newWidth);

public slots:
	void highlight_column(PVHive::PVObserverBase* o);
	void highlight_column(int col);
	void set_section_visible(PVHive::PVObserverBase* o);
	void section_hovered_enter(int col, bool enter);

private:
	QMenu* _ctxt_menu;
	QMenu* _hhead_ctxt_menu;
	QAction* _action_col_sort;
	QAction* _action_col_unique;
	QMenu* _menu_col_count_by;
	QMenu* _menu_col_sum_by;
	QMenu* _menu_col_min_by;
	QMenu* _menu_col_max_by;
	bool _show_ctxt_menu;
	PVRow _ctxt_row;
	PVCol _ctxt_col;
	QString _ctxt_v;
	PVCore::PVArgumentList _ctxt_args;
	PVGuiQt::PVLayerFilterProcessWidget* _ctxt_process;
	QAction* _act_copy;
	QAction* _act_set_color;

	std::unordered_map<uint32_t, uint32_t> _headers_width;
	int _hovered_axis = -1;

	PVWidgets::PVHelpWidget *_help_widget;

private:
	// Observers
	PVHive::PVObserverSignal<Picviz::PVView> _obs;
	PVHive::PVObserverSignal<int> _axis_hover_obs;
	PVHive::PVObserverSignal<PVCol> _axis_clicked_obs;

	// Actor
	PVHive::PVActor<Picviz::PVView> _actor;
};

class PVHorizontalHeaderView : public QHeaderView
{
	Q_OBJECT

public:
	PVHorizontalHeaderView(Qt::Orientation orientation, PVListingView* parent);

signals:
   	void mouse_hovered_section(int index, bool entered);

protected:
	bool event(QEvent *ev) override;
	void paintSection(QPainter *painter, const QRect &rect, int logicalIndex) const override;

private:
	PVGuiQt::PVListingView* listing_view() const { return (PVGuiQt::PVListingView*) parent(); }

private:
	int _index = -1;
};

}

#endif // PVLISTINGVIEW_H
