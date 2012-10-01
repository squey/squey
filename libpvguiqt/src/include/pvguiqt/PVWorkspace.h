/**
 * \file PVWorkspace.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVWORKSPACE_H__
#define __PVGUIQT_PVWORKSPACE_H__

#include <QAction>
#include <QEvent>
#include <QList>
#include <QObject>
#include <QMainWindow>
#include <QToolButton>
#include <QWidget>

#include <picviz/PVSource_types.h>
#include <picviz/PVView_types.h>

namespace Picviz
{
class PVView;
}

Q_DECLARE_METATYPE(Picviz::PVView*)

namespace PVGuiQt
{
class PVListingView;
class PVViewDisplay;

class PVWorkspace : public QMainWindow
{
	Q_OBJECT;

	friend class PVViewDisplay;
public:
	PVWorkspace(Picviz::PVSource* source, QWidget* parent = 0);

	Picviz::PVSource* get_source() const { return _source; }

	PVViewDisplay* add_view_display(Picviz::PVView* view, QWidget* view_display, const QString& name, bool can_be_central_display = true, Qt::DockWidgetArea area = Qt::TopDockWidgetArea);
	PVViewDisplay* set_central_display(Picviz::PVView* view, QWidget* view_widget, const QString& name);
	void set_central_display(PVViewDisplay* view_display);

public:
	PVListingView* create_listing_view(Picviz::PVView_sp view_sp);

private slots:
	void switch_with_central_widget(PVViewDisplay* display_dock = nullptr);
	void add_listing_view(bool central = false);
	void create_parallel_view();
	void create_zoomed_parallel_view(Picviz::PVView* view, int axis_id);
	void show_datatree_view(bool show);
	void check_datatree_button(bool check = false);
	void show_layerstack();
	void hide_layerstack();
	void display_destroyed(QObject* object = 0);

private:
	QList<PVViewDisplay*> _displays;
	Picviz::PVSource* _source;
	QToolBar* _toolbar;
	QAction* _datatree_view_action;
	QToolButton* _layerstack_tool_button;
};

}

#endif /* __PVGUIQT_PVWORKSPACE_H__ */
