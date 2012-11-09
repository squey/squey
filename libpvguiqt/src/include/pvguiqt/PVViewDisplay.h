/**
 * \file PVViewDisplay.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVVIEWDISPLAY_H__
#define __PVGUIQT_PVVIEWDISPLAY_H__

#include <QAction>
#include <QEvent>
#include <QCloseEvent>
#include <QDockWidget>
#include <QFocusEvent>
#include <QContextMenuEvent>

#include <pvhive/PVObserverSignal.h>
#include <pvhive/PVCallHelper.h>

namespace Picviz
{
class PVView;
class PVPlotting;
}

namespace PVGuiQt
{

class PVWorkspaceBase;
class PVWorkspace;
class PVOpenWorkspace;
class DisplaysFocusInEventFilter;

class PVViewDisplay : public QDockWidget
{
	Q_OBJECT;

	friend PVWorkspaceBase;
	friend PVWorkspace;
	friend PVOpenWorkspace;

public:
	~PVViewDisplay() { delete _obs_plotting; }

public:
	Picviz::PVView* get_view() { return _view; }
	void set_view(Picviz::PVView* view) { _view = view; }
	void set_current_view();
	void about_to_be_deleted() { _about_to_be_deleted = true; }

protected:
	bool event(QEvent* event) override;
	void contextMenuEvent(QContextMenuEvent* event) override;
	void closeEvent(QCloseEvent * event) override
	{
		emit display_closed();
		QDockWidget::closeEvent(event);
	}

public slots:
	void dragStarted(bool started);
	void dragEnded();

private slots:
	void plotting_updated();

signals:
	void display_closed();
	void try_automatic_tab_switch();

private:
	void maximize_on_screen(int screen_number);
	void register_view(Picviz::PVView* view);

private:
	PVViewDisplay(Picviz::PVView* view, QWidget* view_widget, std::function<QString()> name, bool can_be_central_widget, bool delete_on_close, PVWorkspaceBase* parent);

private:
	Picviz::PVView* _view;
	std::function<QString()> _name;
	PVWorkspaceBase* _workspace;
	QPoint _press_pt;
	PVHive::PVObserverSignal<Picviz::PVPlotting>*  _obs_plotting = nullptr;
	//PVHive::PVObserver_p<Picviz::PVView> _obs_view;
	bool _about_to_be_deleted = false;

};

}

#endif // #ifndef __PVGUIQT_PVVIEWDISPLAY_H__
