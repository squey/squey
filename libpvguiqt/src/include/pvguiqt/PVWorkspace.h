/**
 * \file PVWorkspace.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVWORKSPACE_H__
#define __PVGUIQT_PVWORKSPACE_H__

class QAction;
class QEvent;
class QToolButton;
class QObject;
class QWidget;
#include <QList>

#include <pvguiqt/PVAxesCombinationDialog.h>
#include <pvguiqt/PVListDisplayDlg.h>

#include <pvdisplays/PVDisplaysContainer.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVObserverSignal.h>

#include <picviz/PVSource_types.h>
#include <picviz/PVView_types.h>

namespace Picviz {
class PVView;
}

Q_DECLARE_METATYPE(Picviz::PVView*)

namespace PVDisplays {
class PVDisplayViewIf;
class PVDisplayViewAxisIf;
class PVDisplayViewZoneIf;
}

namespace PVGuiQt {

class PVListingView;
class PVViewDisplay;

/**
 * \class PVWorkspaceBase
 *
 * \note This class is the base class for workspaces.
 */
class PVWorkspaceBase: public PVDisplays::PVDisplaysContainer
{
	Q_OBJECT

	friend class PVViewDisplay;
	friend class PVViewWidgets;

public:
	typedef PVHive::PVObserverSignal<PVCore::PVDataTreeObjectBase> datatree_obs_t;

private:
		class PVViewWidgets
		{
		public:
			PVGuiQt::PVAxesCombinationDialog* _axes_combination_editor;
			PVViewWidgets(Picviz::PVView* view, PVWorkspaceBase* tab)
			{
				Picviz::PVView_sp view_sp = view->shared_from_this();
				_axes_combination_editor = new PVAxesCombinationDialog(view_sp, tab);
			}
			PVViewWidgets() { _axes_combination_editor = nullptr; }
			~PVViewWidgets() {};

		protected:
			void delete_widgets() { _axes_combination_editor->deleteLater(); }
		};

public:
		PVViewWidgets const& get_view_widgets(Picviz::PVView* view);
		PVAxesCombinationDialog* get_axes_combination_editor(Picviz::PVView* view)
		{
			PVViewWidgets const& widgets = get_view_widgets(view);
			return widgets._axes_combination_editor;
		}

public:
	PVWorkspaceBase(QWidget* parent): PVDisplays::PVDisplaysContainer(parent) {}
	virtual ~PVWorkspaceBase() = 0;

public:
	/*! \brief Create a view display from a widget and add it to the workspace.
	 *
	 *  \param[in] view The underlying PVView.
	 *  \param[in] view_widget The widget displayed by the dock widget.
	 *  \param[in] name The function returning the name of the display based on its type.
	 *  \param[in] can_be_central_widget Specifies if the display can be set as central display.
	 *  \param[in] delete_on_close Specifies if the display is deleted when closed.
	 *  \param[in] area The area of the QDockWidget on the QMainWindow.
	 *
	 *  \return A pointer to the view display.
	 */
	PVViewDisplay* add_view_display(
		Picviz::PVView* view,
		QWidget* view_display,
		std::function<QString()> name,
		bool can_be_central_display = true,
		bool delete_on_close = true,
		Qt::DockWidgetArea area = Qt::TopDockWidgetArea
	);

	/*! \brief Set a widget as the cental view display of the workspace.
	 *
	 *  \param[in] view The underlying PVView.
	 *  \param[in] view_widget The widget displayed by the dock widget.
	 *  \param[in] name The function returning the name of the display based on its type.
	 *  \param[in] delete_on_close Specifies if the display is deleted when closed.
	 *
	 *  \return A pointer to the view display.
	 */
	PVViewDisplay* set_central_display(
		Picviz::PVView* view,
		QWidget* view_widget,
		std::function<QString()> name,
		bool delete_on_close
	);

	/*! \brief Return the workspace located under the mouse.
	 */
	static PVWorkspaceBase* workspace_under_mouse();

public:
	static bool drag_started() { return _drag_started; }
	inline int z_order() { return _z_order_index; }
	void displays_about_to_be_deleted();

protected:
	/*! \brief Keep track of the Z Order of the workspace.
	 *
	 *  \note Used by workspace_under_mouse to disambiguate overlapping workspaces.
	 */
	void changeEvent(QEvent *event) override;

public slots:
	/*! \brief Create the widget used by the view display.
	 *
	 *  \param[in] act The QAction triggering the creation of the widget.
	 */
	void create_view_widget(QAction* act = nullptr) override;

private slots:
	/*! \brief Create the widget used by the view display with axis parameter.
	 *
	 *  \param[in] act The QAction triggering the creation of the widget.
	 */
	void create_view_axis_widget(QAction* act = nullptr) override;

	/*! \brief Create the widget used by the view display with zone parameter.
	 *
	 *  \param[in] act The QAction triggering the creation of the widget.
	 */
	void create_view_zone_widget(QAction* act = nullptr) override;

	/*! \brief Create or display the widget used by the view display.
	 *
	 *  \param[in] act The QAction triggering the creation/display of the widget.
	 */
	void toggle_unique_source_widget(QAction* act = nullptr) override;

	/*! \brief Switch a view display with the central widget.
	 *
	 *  \param[in] display_dock The view display to switch as central widget.
	 *
	 *  \note In order to keep the displays positions, the displays themselves are
	 *        not really switched: instead we switch all of their content (name, colors, observers...)
	 */
	void switch_with_central_widget(PVViewDisplay* display_dock = nullptr);

	/*! \brief Update the internal list of displays when one of them are destroyed.
	 *
	 *  \param[in] object The destroyed view display.
	 *
	 *  \note The internal list of displays is used by toggle_unique_source_widget.
	 */
	void display_destroyed(QObject* object = 0);

signals:
	/*! \brief Signal forwarded when a display is moved in order to detected a potential tab change.
	 */
	void try_automatic_tab_switch();

protected:
	QList<PVViewDisplay*> _displays;
	QList<std::pair<QToolButton*, PVDisplays::PVDisplayViewIf*>> _view_display_if_btns;
	QList<std::pair<QToolButton*, PVDisplays::PVDisplayViewAxisIf*>> _view_axis_display_if_btns;
	QList<std::pair<QToolButton*, PVDisplays::PVDisplayViewZoneIf*>> _view_zone_display_if_btns;
	int _z_order_index = 0;
	static uint64_t _z_order_counter;
	static bool _drag_started;
	QHash<Picviz::PVView const*, PVViewWidgets> _view_widgets;
};

/**
 * \class PVSourceWorkspace
 *
 * \note This class is a PVWorkspaceBase derivation representing workspaces related to a source.
 */
class PVSourceWorkspace : public PVWorkspaceBase
{
	Q_OBJECT

public:
	PVSourceWorkspace(Picviz::PVSource* source, QWidget* parent = 0);

public:
	inline Picviz::PVSource* get_source() const { return _source; }
	inline PVGuiQt::PVListDisplayDlg* get_source_invalid_evts_dlg() const { return _inv_evts_dlg; }

private slots:
	/*! \brief Check if the view count has changed in order to refresh toolbar menus.
	 */
	void update_view_count(PVHive::PVObserverBase* obs_base);

private:
	/*! \brief Refresh toolbar menus to reflect views changes.
	 */
	void refresh_views_menus();

private:
	Picviz::PVSource* _source = nullptr;
	QToolBar* _toolbar;
	std::list<datatree_obs_t> _obs;
	uint64_t _views_count;

	PVGuiQt::PVListDisplayDlg* _inv_evts_dlg;
};

/**
 * \class PVOpenWorkspace
 *
 * \note This class is a PVWorkspaceBase derivation representing open workspaces i.e, not related to any particular source.
 *       Each open workspace has its own current correlation (selected in the list of all available correlations).
 */
class PVOpenWorkspace : public PVWorkspaceBase
{
	Q_OBJECT

public:
	PVOpenWorkspace(QWidget* parent = 0) : PVWorkspaceBase(parent) {}

public:
	void set_correlation(Picviz::PVAD2GView* correlation) { _correlation = correlation; }
	Picviz::PVAD2GView* get_correlation() const { return _correlation; }

private:
	Picviz::PVAD2GView* _correlation = nullptr;
};

}

#endif /* __PVGUIQT_PVWORKSPACE_H__ */
