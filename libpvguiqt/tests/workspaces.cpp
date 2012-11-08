#if 0
/**
 * \file workspaces.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */


#include <pvkernel/core/lambda_connect.h>

#include "workspaces.h"

#include <iostream>
#include <QDateTime>

static bool drag_started = false;

unsigned int CustomMainWindow::zOrderCounter = 0;

CustomMainWindow::CustomMainWindow(QWidget* parent /* = 0*/) : QMainWindow(parent)
{
        setGeometry(
                QStyle::alignedRect(
                        Qt::LeftToRight,
                        Qt::AlignCenter,
                        size(),
                        QApplication::desktop()->availableGeometry()
                ));
}

void CustomMainWindow::dragStarted(bool started)
{
        if(started)
        {
                if(/*CustomDockWidget* dock = */qobject_cast<CustomDockWidget*>(sender())) {
                        drag_started = true;
                }
        }
}

void CustomMainWindow::dragEnded()
{
        drag_started = false;
}

void CustomMainWindow::CreateDockWidgets()
{
        CustomDockWidget* dock_widget = new CustomDockWidget(this);
        dock_widget->setWidget(new QPushButton("Button"));
        dock_widget->setMouseTracking(true);
        //dock_widget->setTitleBarWidget(new QLabel("title"));
        addDockWidget(Qt::LeftDockWidgetArea, dock_widget);

        connect(dock_widget, SIGNAL(topLevelChanged(bool)), this, SLOT(dragStarted(bool)));
        connect(dock_widget, SIGNAL(dockLocationChanged (Qt::DockWidgetArea)), this, SLOT(dragEnded()));
}


void CustomMainWindow::changeEvent(QEvent *event)
{
        QMainWindow::changeEvent(event);

        if (event->type() == QEvent::ActivationChange && isActiveWindow())
        {
                zOrderIndex = ++zOrderCounter;
        }
}

CustomMainWindow* CustomDockWidget::workspace_under_mouse()
{
        CustomMainWindow* main_window_under_mouse = nullptr;
        int z_oder = -1;

        for (QWidget* top_widget : QApplication::topLevelWidgets()) {
                CustomMainWindow* main_window = qobject_cast<CustomMainWindow*>(top_widget);
                if (main_window) {
                        QRect main_global_rect = main_window->geometry();
                        if (main_global_rect.contains(QCursor::pos()) && main_window->z_order() > z_oder) {
                                z_oder = main_window->z_order();
                                main_window_under_mouse = main_window;
                        }
                }
        }

        return main_window_under_mouse;
}

bool CustomDockWidget::event(QEvent* event)
{
        switch (event->type()) {
                case QEvent::MouseMove:
                {
                        QMouseEvent* mouse_event = (QMouseEvent*) event;

                        CustomMainWindow* main_window = workspace_under_mouse();

                        if (main_window) {
                                if (drag_started && main_window && main_window != parent()) {
                                		std::cout << this << "/fake event release" << std::endl;

                                        QMouseEvent fake_event3(QEvent::MouseButtonRelease, mouse_event->pos(), Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
                                        QDockWidget::event(&fake_event3);

                                        std::cout << this << "/remove dock widget + show" << std::endl;
                                        qobject_cast<CustomMainWindow*>(parent())->removeDockWidget(this);
                                        show();

                                        std::cout << this << "/addDockWidget + floating" << std::endl;
                                        main_window->activateWindow();
                                        main_window->addDockWidget(Qt::RightDockWidgetArea, this); // Qt::NoDockWidgetArea yields "QMainWindow::addDockWidget: invalid 'area' argument"
                                        setFloating(true); // We don't want the dock widget to be docked

                                        QPoint curpos = QCursor::pos();
                                        std::cout << this << "/process event: qcursor::pos: " << curpos.x() << "/" << curpos.y() << std::endl;

                                        QCursor::setPos(mapToGlobal(_press_pt));
                                        move(mapToGlobal(_press_pt));

                                        QMouseEvent* fake_event1 = new QMouseEvent(QEvent::MouseButtonPress, _press_pt, Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
                                        QApplication::postEvent(this, fake_event1);

                                        curpos = QCursor::pos();
                                        std::cout << this << "/process event: qcursor::pos: " << curpos.x() << "/" << curpos.y() << std::endl;

                                        QApplication::processEvents(QEventLoop::AllEvents);

                                        curpos = QCursor::pos();
                                        std::cout << this << "/process event: qcursor::pos: " << curpos.x() << "/" << curpos.y() << std::endl;

                                        grabMouse();


                                        return true;
                                }
                        }
                        break;
                }
                case QEvent::MouseButtonPress:
                {
                        QMouseEvent* mouse_event = (QMouseEvent*) event;
                        std::cout << this << "/Press mouse point: " << mouse_event->pos().x() << "/" << mouse_event->pos().y() << std::endl;
                        _press_pt = mouse_event->pos();
                        break;
                }
                case QEvent::MouseButtonRelease:
                {
                        QMouseEvent* mouse_event = (QMouseEvent*) event;
                        std::cout << this << "/Release mouse point: " << mouse_event->pos().x() << "/" << mouse_event->pos().y() << std::endl;
                        break;
                }
                case QEvent::Leave:
                {
                        std::cout << this << "/Mouse leaving" << std::endl;
                        break;
                }
                default:
                {
                        //std::cout << "CustomDockWidget event: " << event->type() << std::endl;
                        break;
                }

        }

        std::cout << this << ": " << event->type() << std::endl;
        QMouseEvent* mouse_event = dynamic_cast<QMouseEvent*>(event);
        if (mouse_event) {
            std::cout << this << "/mouse_event=(" << mouse_event->pos().x() << ";" << mouse_event->pos().y() << ")" << std::endl;
        }

        std::cout << this << "/_press_pt=(" << _press_pt.x() << ";" << _press_pt.y() << ")" << std::endl;

        return QDockWidget::event(event);
}

int main(int argc, char** argv)
{
        QApplication app(argc, argv);

        CustomMainWindow* mw1 = new CustomMainWindow();
        mw1->setWindowTitle("MW1");
        mw1->setCentralWidget(new QLabel("centralWidget"));
        CustomMainWindow* mw2 = new CustomMainWindow();
        mw2->setWindowTitle("MW2");
        mw2->setCentralWidget(new QLabel("centralWidget"));

        mw1->CreateDockWidgets();
        mw2->CreateDockWidgets();

        mw1->show();
        mw2->show();

        return app.exec();
}


#else
/**
 * \file workspaces.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */



#include <pvkernel/core/picviz_intrin.h>

#include <picviz/PVRoot.h>
#include <picviz/PVMapped.h>
#include <picviz/PVPlotted.h>
#include <picviz/PVSource.h>
#include <picviz/PVView.h>

#include <pvhive/PVActor.h>
#include <pvhive/PVCallHelper.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVParallelView.h>

#include <pvguiqt/common.h>
#include <pvguiqt/PVOpenWorkspacesWidget.h>
#include <pvguiqt/PVWorkspace.h>
#include <pvguiqt/PVWorkspacesTabWidget.h>


#include "common.h"
#include "test-env.h"

#include <iostream>

#include <QApplication>
#include <QMainWindow>
#include <QTabWidget>
#include <QStyle>
#include <QDesktopWidget>
#include <QLabel>
#include <QPushButton>


int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " file format" << std::endl;
		return 1;
	}

	PVCore::PVIntrinsics::init_cpuid();
	init_env();

	// Get a Picviz tree from the given file/format
	Picviz::PVRoot_sp root = Picviz::PVRoot::get_root_sp();
	Picviz::PVSource_sp src = get_src_from_file(root, argv[1], argv[2]);
	src->create_default_view();

	Picviz::PVView_p view(src->current_view()->get_parent()->shared_from_this());
	view->process_parent_plotted();

	// Qt app
	QApplication app(argc, argv);

	PVParallelView::common::init_cuda();
	PVGuiQt::common::register_displays();

	// Create our model and view
	root->dump();
	src->dump();

	PVGuiQt::PVSceneWorkspacesTabWidget* workspaces_tab_widget = new PVGuiQt::PVSceneWorkspacesTabWidget(src->get_parent()->shared_from_this());

	PVGuiQt::PVWorkspace* workspace = new PVGuiQt::PVWorkspace(src.get());
	workspaces_tab_widget->addTab(workspace, "Workspace1");
	workspaces_tab_widget->show();

	PVGuiQt::PVOpenWorkspacesWidget* open_workspace = new PVGuiQt::PVOpenWorkspacesWidget(root.get(), NULL);
	open_workspace->show();

	return app.exec();
}

#endif
