/**
 * \file PVWorkspace.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVWORKSPACE_H__
#define __PVGUIQT_PVWORKSPACE_H__

#include <list>
#include <iostream>

#include <QAction>
#include <QEvent>
#include <QList>
#include <QObject>
#include <QMainWindow>
#include <QToolButton>
#include <QWidget>

#include <picviz/PVSource_types.h>
#include <picviz/PVView_types.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVObserverSignal.h>
#include <pvhive/waxes/waxes.h>

#include <pvguiqt/PVAxesCombinationDialog.h>

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

private:
	class PVViewWidgets
	{
		friend class PVWorkspace;
	public:
		PVViewWidgets(Picviz::PVView* view, PVWorkspace* tab)
		{
			Picviz::PVView_sp view_sp = view->shared_from_this();
			pv_axes_combination_editor = new PVAxesCombinationDialog(view_sp, tab);
		}
		PVViewWidgets() { pv_axes_combination_editor = nullptr; /*pv_axes_properties = nullptr;*/ }
		~PVViewWidgets() {};
	protected:
		void delete_widgets() { pv_axes_combination_editor->deleteLater(); }
	public:
		PVGuiQt::PVAxesCombinationDialog* pv_axes_combination_editor;
		//PVAxisPropertiesWidget  *pv_axes_properties;
	};

	friend class PVViewWidgets;

public:
	typedef PVHive::PVObserverSignal<PVCore::PVDataTreeObjectBase> datatree_obs_t;

public:
	PVWorkspace(Picviz::PVSource* source, QWidget* parent = 0);

	static PVWorkspace* workspace_under_mouse();
	static bool drag_started() { return _drag_started; }

	Picviz::PVSource* get_source() const { return _source; }
	PVViewWidgets const& get_view_widgets(Picviz::PVView* view);
	PVAxesCombinationDialog* get_axes_combination_editor(Picviz::PVView* view)
	{
		PVViewWidgets const& widgets = get_view_widgets(view);
		return widgets.pv_axes_combination_editor;
	}

	PVViewDisplay* add_view_display(Picviz::PVView* view, QWidget* view_display, const QString& name, bool can_be_central_display = true, Qt::DockWidgetArea area = Qt::TopDockWidgetArea);
	PVViewDisplay* set_central_display(Picviz::PVView* view, QWidget* view_widget, const QString& name);
	void set_central_display(PVViewDisplay* view_display);

public:
	PVListingView* create_listing_view(Picviz::PVView_sp view_sp);

public:
	Picviz::PVView* get_lib_view();
	Picviz::PVView const* get_lib_view() const;
	inline Picviz::PVSource* get_lib_src() { return _source; }
	inline Picviz::PVSource const* get_lib_src() const { return _source; }
	inline int z_order() { return _z_order_index; }

public slots:
	void switch_with_central_widget(PVViewDisplay* display_dock = nullptr);
	void add_listing_view(bool central = false);
	void create_parallel_view(Picviz::PVView* view = nullptr);
	void create_zoomed_parallel_view();
	void create_zoomed_parallel_view(Picviz::PVView* view, int axis_id);
	void show_datatree_view(bool show);
	void check_datatree_button(bool check = false);
	void create_layerstack(Picviz::PVView* view = nullptr);
	void destroy_layerstack();
	void display_destroyed(QObject* object = 0);
	void update_view_count(PVHive::PVObserverBase* obs_base);
	void emit_try_automatic_tab_switch() { emit try_automatic_tab_switch(); }


signals:
	void try_automatic_tab_switch();

private:
	void refresh_views_menus();

protected:
	void changeEvent(QEvent *event) override;

private:
	QList<PVViewDisplay*> _displays;
	Picviz::PVSource* _source;
	QToolBar* _toolbar;
	QAction* _datatree_view_action;

	QToolButton* _layerstack_tool_button;
	QToolButton* _listing_tool_button;
	QToolButton* _parallel_view_tool_button;
	QToolButton* _zoomed_parallel_view_tool_button;

	std::list<datatree_obs_t> _obs;
	uint64_t _views_count;
	QHash<Picviz::PVView const*, PVViewWidgets> _view_widgets;

	int _z_order_index = 0;
	static uint64_t _z_order_counter;
	static bool _drag_started;
};

}

#endif /* __PVGUIQT_PVWORKSPACE_H__ */
