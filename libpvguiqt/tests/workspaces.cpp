/**
 * \file workspaces.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */


#include <pvkernel/core/lambda_connect.h>

#include "workspaces.h"

#include <iostream>

static bool drag_started = false;

CustomMainWindow::CustomMainWindow()
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

void CustomMainWindow::CreateDockWidgets()
{
	CustomDockWidget* dock_widget = new CustomDockWidget(this);
	dock_widget->setWidget(new QPushButton("Button"));
	addDockWidget(Qt::LeftDockWidgetArea, dock_widget);

	connect(dock_widget, SIGNAL(topLevelChanged(bool)), this, SLOT(dragStarted(bool)));
	//connect(dock_widget, SIGNAL(dockLocationChanged (Qt::DockWidgetArea)), this, SLOT(dragEnded()));
}

bool CustomDockWidget::event(QEvent* event)
{
	switch (event->type()) {
		case QEvent::MouseMove:
		{
			QMouseEvent* mouse_event = (QMouseEvent*) event;

			for (QWidget* top_widget : QApplication::topLevelWidgets()) {
				CustomMainWindow* main_window = qobject_cast<CustomMainWindow*>(top_widget);

				QPoint mouse_global_pos = mouse_event->globalPos();

				if (main_window) {
					QRect main_global_rect = main_window->geometry();

					if (drag_started && main_window && main_window != parent()) {
						if (main_global_rect.contains(mouse_global_pos)) {
							QMouseEvent fake_event3(QEvent::MouseButtonRelease, mouse_event->pos(), Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
							QDockWidget::event(&fake_event3);

							main_window->activateWindow();
							main_window->addDockWidget(Qt::RightDockWidgetArea, this); // Qt::NoDockWidgetArea yields "QMainWindow::addDockWidget: invalid 'area' argument"
							setFloating(true); // We don't want to dock widget to be docked

							QMouseEvent* fake_event_rel = new QMouseEvent(QEvent::MouseButtonRelease, mouse_event->pos(), Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
							QMouseEvent* fake_event1 = new QMouseEvent(QEvent::MouseButtonPress, mouse_event->pos(), Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
							//QMouseEvent* fake_event2 = new QMouseEvent(QEvent::MouseMove, mouse_event->pos(), Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);

							//QApplication::postEvent(this, fake_event_rel);
							//QApplication::postEvent(this, fake_event1);
							QDockWidget::event(fake_event_rel);
							QDockWidget::event(fake_event1);

							/*QMouseEvent fake_event2(QEvent::DragMove, mouse_event->pos(), Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
							QApplication::sendEvent(this, &fake_event1);
							QApplication::sendEvent(this, &fake_event2);*/

							drag_started = false;
							return true;
						}
					}
				}
			}
			break;
		}
		default:
		{
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
