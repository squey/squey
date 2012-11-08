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

#include <pvdisplays/PVDisplaysContainer.h>

#include <pvguiqt/PVAxesCombinationDialog.h>

namespace Picviz {
class PVView;
}

namespace PVDisplays {
class PVDisplayViewIf;
class PVDisplayViewAxisIf;
}

Q_DECLARE_METATYPE(Picviz::PVView*)

namespace PVGuiQt {
class PVListingView;
class PVViewDisplay;

class PVWorkspaceBase: public PVDisplays::PVDisplaysContainer
{
	Q_OBJECT

		friend class PVViewDisplay;

private:
		class PVViewWidgets
		{
		public:
			PVViewWidgets(Picviz::PVView* view, PVWorkspaceBase* tab)
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
public:
		PVViewWidgets const& get_view_widgets(Picviz::PVView* view);
		PVAxesCombinationDialog* get_axes_combination_editor(Picviz::PVView* view)
		{
			PVViewWidgets const& widgets = get_view_widgets(view);
			return widgets.pv_axes_combination_editor;
		}

		friend class PVViewWidgets;

public:
	PVWorkspaceBase(QWidget* parent): PVDisplays::PVDisplaysContainer(parent) {}
	virtual ~PVWorkspaceBase() = 0;

public:
	typedef PVHive::PVObserverSignal<PVCore::PVDataTreeObjectBase> datatree_obs_t;

	Picviz::PVView* current_view() const { return _current_view; }
	void set_current_view(Picviz::PVView* view) { _current_view = view; }

	static PVWorkspaceBase* workspace_under_mouse();
	static bool drag_started() { return _drag_started; }

	PVViewDisplay* add_view_display(Picviz::PVView* view, QWidget* view_display, const QString& name, bool can_be_central_display = true, bool delete_on_close = true, Qt::DockWidgetArea area = Qt::TopDockWidgetArea);
	PVViewDisplay* set_central_display(Picviz::PVView* view, QWidget* view_widget, const QString& name, bool delete_on_close);
	void set_central_display(PVViewDisplay* view_display);
	inline int z_order() { return _z_order_index; }

public slots:
	void switch_with_central_widget(PVViewDisplay* display_dock = nullptr);
	void display_destroyed(QObject* object = 0);

public slots:
	void create_view_widget() override;
	void create_view_axis_widget() override; 
	void toggle_unique_source_widget() override;

signals:
	void try_automatic_tab_switch();

protected:
	void changeEvent(QEvent *event) override;

protected:
	QList<PVViewDisplay*> _displays;
	QList<std::pair<QToolButton*, PVDisplays::PVDisplayViewIf*>> _view_display_if_btns;
	QList<std::pair<QToolButton*, PVDisplays::PVDisplayViewAxisIf*>> _view_axis_display_if_btns;
	int _z_order_index = 0;
	static uint64_t _z_order_counter;
	static bool _drag_started;
	QHash<Picviz::PVView const*, PVViewWidgets> _view_widgets;

	Picviz::PVView* _current_view = nullptr;
};

class PVWorkspace : public PVWorkspaceBase
{
	Q_OBJECT

public:
	inline Picviz::PVSource* get_source() const { return _source; }

public:
	PVWorkspace(Picviz::PVSource* source, QWidget* parent = 0);

public:
	PVListingView* create_listing_view(Picviz::PVView_sp view_sp);

public slots:
	void update_view_count(PVHive::PVObserverBase* obs_base);

private:
	void refresh_views_menus();

private:
	Picviz::PVSource* _source = nullptr;
	QToolBar* _toolbar;

	std::list<datatree_obs_t> _obs;
	uint64_t _views_count;
};


class PVOpenWorkspace : public PVWorkspaceBase
{
	Q_OBJECT
public:
	PVOpenWorkspace(QWidget* parent = 0);
	void set_correlation_index(int index) { _correlation_index = index; }
	int get_correlation_index() const { return _correlation_index; }

private:
	int _correlation_index = 0;
};

}

#endif /* __PVGUIQT_PVWORKSPACE_H__ */
