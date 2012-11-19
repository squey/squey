/**
 * \file PVViewDisplay.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVVIEWDISPLAY_H__
#define __PVGUIQT_PVVIEWDISPLAY_H__

#include <pvhive/PVObserverSignal.h>
#include <pvhive/PVCallHelper.h>

class QString;
class QPoint;
class QWidget;
#include <QDockWidget>

namespace Picviz
{
class PVView;
}

namespace PVGuiQt
{

class PVWorkspaceBase;
class PVSourceWorkspace;
class PVOpenWorkspace;

/**
 * \class PVViewDisplay
 *
 * \note This class is a dockable wrapper for graphical view representations.
 */
class PVViewDisplay : public QDockWidget
{
	Q_OBJECT;

	friend PVWorkspaceBase;
	friend PVSourceWorkspace;
	friend PVOpenWorkspace;

public:
	~PVViewDisplay() { delete _obs_plotting; }

public:
	/*! \brief Call Picviz::PVRoot::select_view through the Hive.
	 * This is called by the application level events filter DisplaysFocusInEventFilter on PVViewDisplay QEvent::FocusIn events.
	 */
	void set_current_view();

public:
	Picviz::PVView* get_view() { return _view; }
	void set_view(Picviz::PVView* view) { _view = view; }
	void about_to_be_deleted() { _about_to_be_deleted = true; }

protected:
	/*! \brief Filter events to allow a PVViewDisplay to be docked inside any other PVWorkspace.
	 */
	bool event(QEvent* event) override;

	/*! \brief Create the view display right click menu.
	 */
	void contextMenuEvent(QContextMenuEvent* event) override;

private slots:
	/*! \brief Store the state of the drag&drop operation.
	 */
	void drag_started(bool started);

	/*! \brief Create the view display right click menu.
	 */
	void plotting_updated();

signals:
	/*! \brief Signal emited when the display is moved in order to detected a potential tab change.
	 */
	void try_automatic_tab_switch();

private:
	/*! \brief Maximize a view display on a given screen.
	 */
	void maximize_on_screen(int screen_number);

	/*! \brief Register the view to handle several events.
	 */
	void register_view(Picviz::PVView* view);

private:
	/*! \brief Creates a view display.
	 *  \param[in] view The underlying PVView.
	 *  \param[in] view_widget The widget displayed by the dock widget.
	 *  \param[in] name The function returning the name of the display based on its type.
	 *  \param[in] can_be_central_widget Specifies if the display can be set as central display.
	 *  \param[in] delete_on_close Specifies if the display is deleted when closed.
	 *  \param[in] workspace The parent workspace.
	 *
	 *  \note this constructor is intended to be called only by PVWorkspace, hence the private visibility.
	 */
	PVViewDisplay(
		Picviz::PVView* view,
		QWidget* view_widget,
		std::function<QString()> name,
		bool can_be_central_widget,
		bool delete_on_close,
		PVWorkspaceBase* parent
	);

private:
	Picviz::PVView* _view;
	std::function<QString()> _name;
	PVWorkspaceBase* _workspace;
	QPoint _press_pt;
	PVHive::PVObserverSignal<Picviz::PVPlotting>*  _obs_plotting = nullptr;
	PVHive::PVObserver_p<Picviz::PVView> _obs_view;
	bool _about_to_be_deleted = false;

};

}

#endif // #ifndef __PVGUIQT_PVVIEWDISPLAY_H__
