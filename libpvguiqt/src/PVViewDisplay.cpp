//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <functional>

#include <QGuiApplication>
#include <QAbstractScrollArea>
#include <QAction>
#include <QApplication>
#include <QContextMenuEvent>
#include <QEvent>
#include <QMenu>
#include <QMouseEvent>
#include <QScrollBar>
#include <QScreen>

#include <pvguiqt/PVViewDisplay.h>
#include <pvguiqt/PVWorkspace.h>
#include <pvguiqt/PVWorkspacesTabWidget.h>

#include <inendi/PVView.h>
#include <inendi/PVRoot.h>
#include <inendi/PVPlotted.h>

PVGuiQt::PVViewDisplay::PVViewDisplay(Inendi::PVView* view,
                                      QWidget* view_widget,
                                      QString name,
                                      bool can_be_central_widget,
                                      bool delete_on_close,
                                      PVWorkspaceBase* workspace)
    : QDockWidget((QWidget*)workspace)
    , _view(view)
    , _name(name)
    , _workspace(workspace)
    , _can_be_central_widget(can_be_central_widget)
{
	setWidget(view_widget);
	setWindowTitle(_name);

	setFocusPolicy(Qt::StrongFocus);
	view_widget->setFocusPolicy(Qt::StrongFocus);

	QAbstractScrollArea* scroll_area = dynamic_cast<QAbstractScrollArea*>(view_widget);
	if (scroll_area) {
		scroll_area->verticalScrollBar()->setObjectName("verticalScrollBar_of_PVListingView");
		scroll_area->horizontalScrollBar()->setObjectName("horizontalScrollBar_of_PVListingView");
	}

	if (view) {
		// Set view color
		QColor view_color = view->get_color();
		QPalette Pal(palette());
		Pal.setColor(QPalette::Window, view_color);
		setAutoFillBackground(true);
		setPalette(Pal);
	}

	if (delete_on_close) {
		setAttribute(Qt::WA_DeleteOnClose, true);
	}

	connect(this, &QDockWidget::topLevelChanged, this, &PVViewDisplay::drag_started);
	connect(this, &QDockWidget::dockLocationChanged, this, &PVViewDisplay::drag_ended);
	connect(view_widget, &QObject::destroyed, this, &QWidget::close);

	_screenSignalMapper = new QSignalMapper(this);
	connect(_screenSignalMapper, SIGNAL(mapped(int)), this, SLOT(maximize_on_screen(int)));

	register_view(view);
}

void PVGuiQt::PVViewDisplay::register_view(Inendi::PVView* view)
{
	if (view) {

		view->get_parent<Inendi::PVPlotted>()._plotted_updated.connect(
		    sigc::mem_fun(this, &PVGuiQt::PVViewDisplay::plotting_updated));
	}
}

bool PVGuiQt::PVViewDisplay::event(QEvent* event)
{
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
	case QEvent::MouseMove: {
		if (PVGuiQt::PVSourceWorkspace::_drag_started) {
			Q_EMIT try_automatic_tab_switch();

			QMouseEvent* mouse_event = (QMouseEvent*)event;
			PVWorkspaceBase* workspace = PVGuiQt::PVSourceWorkspace::workspace_under_mouse();

			// If we are over a new workspace...
			if (workspace && workspace != parent()) {

				QMouseEvent* fake_mouse_release =
				    new QMouseEvent(QEvent::MouseButtonRelease, mapFromGlobal(mouse_event->pos()), mouse_event->pos(), Qt::LeftButton,
				                    Qt::LeftButton, Qt::NoModifier);
				QApplication::postEvent(this, fake_mouse_release);
				QApplication::processEvents(QEventLoop::AllEvents);

				qobject_cast<PVWorkspaceBase*>(parent())->removeDockWidget(this);
				show();

				workspace->activateWindow();
				workspace->addDockWidget(Qt::RightDockWidgetArea,
				                         this); // Qt::NoDockWidgetArea yields
				                                // "QMainWindow::addDockWidget: invalid
				                                // 'area' argument"
				if (!isFloating()) {
					setFloating(true); // We don't want the dock widget to be docked right now
				}

				_workspace = workspace;

				disconnect(this, SIGNAL(try_automatic_tab_switch()), 0, 0);
				connect(this, &PVViewDisplay::try_automatic_tab_switch, workspace,
				        &PVWorkspaceBase::try_automatic_tab_switch);

				QCursor::setPos(mapToGlobal(_press_pt));
				move(mapToGlobal(_press_pt));

				QMouseEvent* fake_mouse_press =
				    new QMouseEvent(QEvent::MouseButtonPress, mapToGlobal(_press_pt), _press_pt, Qt::LeftButton,
				                    Qt::LeftButton, Qt::NoModifier);
				QApplication::postEvent(this, fake_mouse_press);

				QApplication::processEvents(QEventLoop::AllEvents);

				QCursor::setPos(mapToGlobal(_press_pt));

				grabMouse();

				return true;
			}
		}
		break;
	}
	case QEvent::MouseButtonPress: {
		QMouseEvent* mouse_event = (QMouseEvent*)event;
		if (mouse_event->button() == Qt::LeftButton) {
			_press_pt = mouse_event->pos();
			drag_started(true);
		}
		break;
	}
	case QEvent::MouseButtonRelease: {
		QMouseEvent* mouse_event = (QMouseEvent*)event;
		if (mouse_event->button() == Qt::LeftButton) {
			drag_started(false);
		}
		break;
	}
	case QEvent::Move: {
		break;
	}
	default: {
		break;
	}
	}
	return QDockWidget::event(event);
}

void PVGuiQt::PVViewDisplay::drag_started(bool started)
{
	if (started) {
		if (qobject_cast<PVViewDisplay*>(sender())) {
			PVGuiQt::PVWorkspaceBase::_drag_started = true;
		}
	}
	_state = EState::CAN_MAXIMIZE;
}

void PVGuiQt::PVViewDisplay::drag_ended()
{
	PVGuiQt::PVWorkspaceBase::_drag_started = false;
	_state = EState::HIDDEN;
}

void PVGuiQt::PVViewDisplay::contextMenuEvent(QContextMenuEvent* event)
{
	// FIXME: QApplication::desktop()->screenNumber() indexes are not necessarily
	// ordered from left to right...
	bool add_menu = true;
	add_menu &= _workspace->centralWidget() != this;
	add_menu &= !widget()->isAncestorOf(childAt(event->pos()));

	if (add_menu) {
		QMenu* ctxt_menu = new QMenu(this);

		if (_can_be_central_widget) {
			// Set as central display
			QAction* switch_action = new QAction(tr("Set as central display"), this);
			connect(switch_action, SIGNAL(triggered(bool)), (QWidget*)_workspace,
			        SLOT(switch_with_central_widget()));
			ctxt_menu->addAction(switch_action);
		}

		QScreen* screen = QGuiApplication::screenAt(rect().topLeft());
		int screen_number = QGuiApplication::screens().indexOf(screen);

		// Maximize & Restore
		if (_state == EState::CAN_MAXIMIZE) {
			QAction* maximize_action = new QAction(tr("Maximize"), this);
			connect(maximize_action, SIGNAL(triggered(bool)), _screenSignalMapper, SLOT(map()));
			_screenSignalMapper->setMapping(maximize_action, screen_number);
			ctxt_menu->addAction(maximize_action);
		} else if (_state == EState::CAN_RESTORE) {
			QAction* restore_action = new QAction(tr("Restore"), this);
			connect(restore_action, &QAction::triggered, this, &PVViewDisplay::restore);
			ctxt_menu->addAction(restore_action);
		}

		// Maximize on left monitor
		if (screen_number > 0) {
			QAction* maximize_left_action = new QAction(tr("<< Maximize on left screen"), this);
			connect(maximize_left_action, SIGNAL(triggered(bool)), _screenSignalMapper,
			        SLOT(map()));
			_screenSignalMapper->setMapping(maximize_left_action, screen_number - 1);
			ctxt_menu->addAction(maximize_left_action);
		}

		// Maximize on right monitor
		if (screen_number < QGuiApplication::screens().size() - 1) {
			QAction* maximize_right_action = new QAction(tr(">> Maximize on right screen"), this);
			connect(maximize_right_action, SIGNAL(triggered(bool)), _screenSignalMapper,
			        SLOT(map()));
			_screenSignalMapper->setMapping(maximize_right_action, screen_number + 1);
			ctxt_menu->addAction(maximize_right_action);
		}

		ctxt_menu->popup(QCursor::pos());
	}
}

void PVGuiQt::PVViewDisplay::maximize_on_screen(int screen_number)
{
	_width = width();
	_height = height();
	_x = x();
	_y = y();

	QScreen* screen = QGuiApplication::screenAt(rect().topLeft());
	bool can_restore = QGuiApplication::screens().indexOf(screen) == screen_number;

	QRect screenres = QGuiApplication::screens()[screen_number]->geometry();

	// JBL: You may be wondering why the hell I am messing so much with the
	// floating state of the widget,
	//      well, this is to workaround a Qt bug preventing it to be moved on the
	//      proper screen...
	//      So please, don't try to "optimize" this weird methode because you
	//      would surely break something!
	//      (Note: A test case is being written in order to open a bug report)
	if (!isFloating()) {
		setFloating(true);
	}
	resize(screenres.width(), screenres.height());
	move(QPoint(screenres.x(), screenres.y()));

	if (can_restore) {
		_state = EState::CAN_RESTORE;
	} else {
		_state = EState::HIDDEN;
	}
}

void PVGuiQt::PVViewDisplay::restore()
{
	resize(_width, _height);
	move(_x, _y);
	_state = EState::CAN_MAXIMIZE;
}

void PVGuiQt::PVViewDisplay::set_current_view()
{
	if (_view) {
		_view->get_parent<Inendi::PVRoot>().select_view(*_view);
	}
}

void PVGuiQt::PVViewDisplay::plotting_updated(QList<PVCol> const& /*cols_updated*/)
{
	setWindowTitle(_name);
}
