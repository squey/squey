/**
 * \file PVWorkspace.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVWORKSPACE_H__
#define __PVGUIQT_PVWORKSPACE_H__

#include <QAction>
#include <QList>
#include <QObject>
#include <QMainWindow>
#include <QWidget>

#include <picviz/PVSource_types.h>

namespace Picviz
{
class PVView;
}

Q_DECLARE_METATYPE(Picviz::PVView*)

namespace PVGuiQt
{

class PVViewDisplay;

class PVWorkspace : public QMainWindow
{
	Q_OBJECT;

	friend class PVViewDisplay;
public:
	PVWorkspace(Picviz::PVSource_sp, QWidget* parent = 0);

	PVViewDisplay* add_view_display(QWidget* view_display, const QString& name, bool can_be_central_display = true);
	PVViewDisplay* set_central_display(QWidget* view_widget, const QString& name);

public slots:
	void switch_with_central_widget(PVViewDisplay* display_dock = nullptr);
	void create_listing_view();
	void create_parallel_view();
	void create_zoomed_parallel_view(Picviz::PVView* view, int axis_id);
	void show_datatree_view(bool show);
	void check_datatree_button(bool check = false);
	void display_destroyed(QObject* object = 0);

private:
	QList<PVViewDisplay*> _displays;
	Picviz::PVSource_sp _source;
	QToolBar* _toolbar;
	QAction* _datatree_view_action;
};

}

#endif /* __PVGUIQT_PVWORKSPACE_H__ */
