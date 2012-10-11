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
				std::cout << "Z Order=" << main_window->z_order() << std::endl;

				if (drag_started && main_window && main_window != parent()) {
					QMouseEvent fake_event3(QEvent::MouseButtonRelease, mouse_event->pos(), Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
					QDockWidget::event(&fake_event3);

					qobject_cast<CustomMainWindow*>(parent())->removeDockWidget(this);
					show();

					main_window->activateWindow();
					main_window->addDockWidget(Qt::RightDockWidgetArea, this); // Qt::NoDockWidgetArea yields "QMainWindow::addDockWidget: invalid 'area' argument"
					setFloating(true); // We don't want the dock widget to be docked

					QCursor::setPos(mapToGlobal(_press_pt));
					move(mapToGlobal(_press_pt));

					QMouseEvent* fake_event1 = new QMouseEvent(QEvent::MouseButtonPress, _press_pt, Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
					QApplication::postEvent(this, fake_event1);


					QApplication::processEvents(QEventLoop::AllEvents);

					grabMouse();


					return true;
				}
			}
			break;
		}
		case QEvent::MouseButtonPress:
		{
			QMouseEvent* mouse_event = (QMouseEvent*) event;
			std::cout << "Press mouse point: " << mouse_event->pos().x() << "/" << mouse_event->pos().y() << std::endl;
			_press_pt = mouse_event->pos();
			break;
		}
		case QEvent::MouseButtonRelease:
		{
			QMouseEvent* mouse_event = (QMouseEvent*) event;
			std::cout << "Release mouse point: " << mouse_event->pos().x() << "/" << mouse_event->pos().y() << std::endl;
			break;
		}
		case QEvent::Leave:
		{
			std::cout << "Mouse leaving" << std::endl;
			break;
		}
		default:
		{
			//std::cout << "CustomDockWidget event: " << event->type() << std::endl;
			break;
		}

	}
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
