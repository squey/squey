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
#include <pvguiqt/PVDockWidgetTitleBar.h>

#include <squey/PVView.h>
#include <squey/PVRoot.h>
#include <squey/PVPlotted.h>

PVGuiQt::PVViewDisplay::PVViewDisplay(Squey::PVView* view,
                                      QWidget* view_widget,
                                      bool can_be_central_widget,
                                      bool has_help_page,
                                      bool delete_on_close,
                                      PVWorkspaceBase* workspace)
    : QDockWidget((QWidget*)workspace)
    , _view(view)
    , _workspace(workspace)
    , _can_be_central_widget(can_be_central_widget)
    , _has_help_page(has_help_page)
{
	setWidget(view_widget);

	setFocusPolicy(Qt::StrongFocus);
	view_widget->setFocusPolicy(Qt::StrongFocus);

	setTitleBarWidget(new PVDockWidgetTitleBar(view, view_widget, has_help_page, this));
	setWindowTitle(view_widget->windowTitle());

	if (view) {
		// Set view color
		// Workaround for setting color only to the title bar
		QColor view_color = view->get_color();
		QPalette child_widget_palette = palette();
		QPalette title_bar_palette = palette();
		title_bar_palette.setColor(QPalette::Window, view_color);
		setAutoFillBackground(true);
		setPalette(title_bar_palette);
		view_widget->setAutoFillBackground(true);
		view_widget->setPalette(child_widget_palette);
	}

	if (delete_on_close) {
		setAttribute(Qt::WA_DeleteOnClose, true);
	}

	connect(view_widget, &QObject::destroyed, this, &QWidget::close);
}

void PVGuiQt::PVViewDisplay::contextMenuEvent(QContextMenuEvent* event)
{
	// FIXME: QApplication::desktop()->screenNumber() indexes are not necessarily
	// ordered from left to right...
	bool add_menu = true;
	add_menu &= _workspace->centralWidget() != this;
	add_menu &= !widget()->isAncestorOf(childAt(event->pos()));

	if (add_menu) {
		auto* ctxt_menu = new QMenu(this);

		if (_can_be_central_widget) {
			// Set as central display
			auto* switch_action = new QAction(tr("Set as central display"), this);
			connect(switch_action, SIGNAL(triggered(bool)), (QWidget*)_workspace,
			        SLOT(switch_with_central_widget()));
			ctxt_menu->addAction(switch_action);
		}

		// Maximize & Restore
		if (isFloating()) {
			auto* restore_action = new QAction(tr("Restore docking"), this);
			connect(restore_action, &QAction::triggered, [this] { setFloating(false); });
			ctxt_menu->addAction(restore_action);
		}
		if (_state == EState::CAN_RESTORE) {
			auto* restore_action = new QAction(tr("Restore floating"), this);
			connect(restore_action, &QAction::triggered, this, &PVViewDisplay::restore);
			ctxt_menu->addAction(restore_action);
		}

		/*
		    Extract from https://doc.qt.io/qt-6/qscreen.html#availableGeometry-prop Qt6.4
		        Note, on X11 this will return the true available geometry only on systems with one
		   monitor and if window manager has set _NET_WORKAREA atom. In all other cases this is
		   equal to geometry(). This is a limitation in X11 window manager specification.
		*/
		const bool X11_bug = QGuiApplication::primaryScreen()->availableGeometry() ==
		                     QGuiApplication::primaryScreen()->geometry();

		for (int i = 0; i < QGuiApplication::screens().size(); ++i) {
			auto* screen_i = QGuiApplication::screens()[i];
			char const* action_text = (screen() == screen_i) ? "Maximize on screen %n (current)"
			                                                 : "Maximize on screen %n";
			auto* maximize_action = new QAction(tr(action_text, "", i), this);
			if (_workspace->screen() == screen_i ||
			    (screen() == screen_i && _state != EState::CAN_MAXIMIZE) ||
			    (X11_bug && screen_i == QGuiApplication::primaryScreen())) {
				maximize_action->setEnabled(false);
			} else {
				connect(maximize_action, &QAction::triggered, [this, i] {
					if (QGuiApplication::screens().size() > i) {
						maximize_on_screen(QGuiApplication::screens()[i]);
					}
				});
			}
			ctxt_menu->addAction(maximize_action);
		}

		ctxt_menu->popup(QCursor::pos());
	}
}

void PVGuiQt::PVViewDisplay::maximize_on_screen(QScreen* screen)
{
	if (_state == EState::CAN_MAXIMIZE) {
		_width = width();
		_height = height();
		_x = x();
		_y = y();
	}

	QRect screenres = screen->availableGeometry();

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

	if (_state != EState::HIDDEN) {
		_state = EState::CAN_RESTORE;
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
		_view->get_parent<Squey::PVRoot>().select_view(*_view);
	}
}

PVGuiQt::PVDockWidgetTitleBar* PVGuiQt::PVViewDisplay::titlebar_widget()
{
	return static_cast<PVDockWidgetTitleBar*>(titleBarWidget());
}

void PVGuiQt::PVViewDisplay::set_help_page_visible(bool visible)
{
	_has_help_page = visible;
	titlebar_widget()->set_help_page_visible(_has_help_page);
}

void PVGuiQt::PVViewDisplay::setWindowTitle(const QString& window_title)
{
	titlebar_widget()->set_window_title(window_title);
	QDockWidget::setWindowTitle(window_title); // ?
}