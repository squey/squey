/**
 * \file PVViewDisplay.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <QAbstractScrollArea>
#include <QApplication>
#include <QDesktopWidget>
#include <QMenu>
#include <QMouseEvent>
#include <QScrollBar>

#include <pvkernel/core/lambda_connect.h>

#include <pvguiqt/PVViewDisplay.h>
#include <pvguiqt/PVWorkspace.h>
#include <pvguiqt/PVWorkspacesTabWidget.h>

#include <picviz/PVView.h>

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVHive.h>
#include <pvhive/waxes/waxes.h>

#include <X11/Xlib.h>
#include <QX11Info>


PVGuiQt::PVViewDisplay::PVViewDisplay(Picviz::PVView* view, QWidget* view_widget, const QString& name, bool can_be_central_widget, PVWorkspaceBase* workspace) :
	QDockWidget((QWidget*)workspace),
	_view(view),
	_workspace(workspace)
{
	setFocusPolicy(Qt::StrongFocus);
	setWidget(view_widget);
	setWindowTitle(name);

	view_widget->installEventFilter(new FocusInEventFilter(this));
	view_widget->setFocusPolicy(Qt::StrongFocus);

	QAbstractScrollArea* scroll_area = dynamic_cast<QAbstractScrollArea*>(view_widget);
	if (scroll_area) {
		scroll_area->verticalScrollBar()->setObjectName("verticalScrollBar_of_PVListingView");
		scroll_area->horizontalScrollBar()->setObjectName("horizontalScrollBar_of_PVListingView");
	}

	if (view) {
		QColor view_color = view->get_color();
		setStyleSheet(QString("QDockWidget::title {background: %1;} QDockWidget { background: %2;} ").arg(view_color.name()).arg(view_color.name()));
		setFocusPolicy(Qt::StrongFocus);
	}

	if (can_be_central_widget) {
		setAttribute(Qt::WA_DeleteOnClose, true);
	}

	connect(this, SIGNAL(topLevelChanged(bool)), this, SLOT(dragStarted(bool)));
	connect(this, SIGNAL(dockLocationChanged (Qt::DockWidgetArea)), this, SLOT(dragEnded()));
}

bool PVGuiQt::PVViewDisplay::event(QEvent* event)
{
// Allow PVViewDisplay to be docked inside any other PVWorkspace
//                ,
//          ._  \/, ,|_
//          -\| \|;|,'_
//          `_\|\|;/-.
//           `_\\|/._
//          ,'__   __`.
//         / /_ | | _\ \  ┌─────────────────────────────
//        / ((o)| |(o)) \ | Voodoo magic begins here...
//        |  `--/ \--'  | └─────────────────────────────
//  ,--.   `.   '-'   ,'  /
// (O..O)    `.uuuuu,'   y
//  \==/     _|nnnnn|_
// .'||`. ,-' \_____/ `-.
//  _||,-'      | |      `.
// (__)  _,-.   ; |   .'.  `.
// (___)'   |__/___\__|  \(__)
// (__)     :::::::::::  (___)
//   ||    :::::::::::::  (__)
//   ||    :::::::::::::
//        __|   | | _ |__
//       (_(_(_,' '._)_)_)
//
	switch (event->type()) {
		case QEvent::MouseMove:
		{
			if (PVGuiQt::PVWorkspace::_drag_started) {
				emit try_automatic_tab_switch();

				QMouseEvent* mouse_event = (QMouseEvent*) event;
				PVWorkspaceBase* workspace = PVGuiQt::PVWorkspace::workspace_under_mouse();

				if (workspace) {

					if (workspace != parent()) {

						QMouseEvent* fake_mouse_release = new QMouseEvent(QEvent::MouseButtonRelease, mouse_event->pos(), Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
						QApplication::postEvent(this, fake_mouse_release);
						QApplication::processEvents(QEventLoop::AllEvents);

						qobject_cast<PVWorkspaceBase*>(parent())->removeDockWidget(this);
						show();

						workspace->activateWindow();
						workspace->addDockWidget(Qt::RightDockWidgetArea, this); // Qt::NoDockWidgetArea yields "QMainWindow::addDockWidget: invalid 'area' argument"
						setFloating(true); // We don't want the dock widget to be docked right now

						QCursor::setPos(mapToGlobal(_press_pt));
						move(mapToGlobal(_press_pt));

						XSync(QX11Info::display(), false);

						QMouseEvent* fake_mouse_press = new QMouseEvent(QEvent::MouseButtonPress, _press_pt, Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
						QApplication::postEvent(this, fake_mouse_press);


						QApplication::processEvents(QEventLoop::AllEvents);

						QCursor::setPos(mapToGlobal(_press_pt));

						grabMouse();

						return true;
					}
				}
			}
			break;
		}
		case QEvent::MouseButtonPress:
		{
			QMouseEvent* mouse_event = (QMouseEvent*) event;
			_press_pt = mouse_event->pos();
			break;
		}
		case QEvent::MouseButtonRelease:
		{
			PVGuiQt::PVWorkspace::_drag_started = false;
			break;
		}
		case QEvent::Move:
		{
			PVGuiQt::PVWorkspace::_drag_started = true;
			break;
		}
		default:
		{
			break;
		}

	}
	return QDockWidget::event(event);
}

void PVGuiQt::PVViewDisplay::dragStarted(bool started)
{
	if(started)
	{
		if(qobject_cast<PVViewDisplay*>(sender())) {
			PVGuiQt::PVWorkspaceBase::_drag_started = true;
		}
	}
}

void PVGuiQt::PVViewDisplay::dragEnded()
{
	PVGuiQt::PVWorkspace::_drag_started = false;
}

void PVGuiQt::PVViewDisplay::contextMenuEvent(QContextMenuEvent* event)
{
	bool add_menu = true;
	add_menu &= _workspace->centralWidget() != this;
	add_menu &=  !widget()->isAncestorOf(childAt(event->pos()));

	if (add_menu) {
		QMenu* ctxt_menu = new QMenu(this);

		// Set as central display
		QAction* switch_action = new QAction(tr("Set as central display"), this);
		connect(switch_action, SIGNAL(triggered(bool)), (QWidget*)_workspace, SLOT(switch_with_central_widget()));
		ctxt_menu->addAction(switch_action);

		std::cout << "QApplication::desktop()->screenNumber()" << QApplication::desktop()->screenNumber(this) << std::endl;

		// Maximize on left screen
		int screen_number = QApplication::desktop()->screenNumber(this);
		if (screen_number > 0) {
			QAction* maximize_right_action = new QAction(tr(">> Maximize on right screen"), this);
			::connect(maximize_right_action, SIGNAL(triggered(bool)), [=]{maximize_on_screen(screen_number-1);});
			ctxt_menu->addAction(maximize_right_action);
		}

		if (screen_number < QApplication::desktop()->screenCount()-1) {
			QAction* maximize_left_action = new QAction(tr("<< Maximize on left screen"), this);
			::connect(maximize_left_action, SIGNAL(triggered(bool)), [=]{maximize_on_screen(screen_number+1);});
			ctxt_menu->addAction(maximize_left_action);
		}

		ctxt_menu->popup(QCursor::pos());
	}
}

void PVGuiQt::PVViewDisplay::maximize_on_screen(int screen_number)
{
	QRect screenres = QApplication::desktop()->screenGeometry(screen_number);
	setFloating(true);
	move(QPoint(screenres.x(), screenres.y()));
	resize(screenres.width(), screenres.height());
	show();
}

void PVGuiQt::PVViewDisplay::set_current_view()
{
	if (_view) {
		auto source = _view->get_parent<Picviz::PVSource>()->shared_from_this();
		PVHive::call<FUNC(Picviz::PVSource::select_view)>(source, *_view);
	}
}
