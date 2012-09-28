/**
 * \file workspaces.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvguiqt/PVWorkspace.h>

#include <iostream>

#include <QApplication>
#include <QMainWindow>
#include <QTabWidget>
#include <QStyle>
#include <QDesktopWidget>
#include <QLabel>
#include <QPushButton>

class CustomMainWindow : public QMainWindow
{
public:

	CustomMainWindow()
	{
		setMinimumSize(500, 600);

		setGeometry(
		    QStyle::alignedRect(
		        Qt::LeftToRight,
		        Qt::AlignCenter,
		        size(),
		        qApp->desktop()->availableGeometry()
		    ));
	}
};

int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	QMainWindow* mw = new CustomMainWindow();

	QWidget* main_widget = new QWidget();

	PVGuiQt::PVWorkspace* workspace1 = new PVGuiQt::PVWorkspace();
	workspace1->setTabPosition(Qt::TopDockWidgetArea, QTabWidget::North);

	PVGuiQt::PVWorkspace* workspace2 = new PVGuiQt::PVWorkspace();
	workspace2->setTabPosition(Qt::TopDockWidgetArea, QTabWidget::North);

	QTabWidget* tab_widget = new QTabWidget(mw);
	tab_widget->resize(640, 600);
	tab_widget->setTabsClosable(true);

	tab_widget->addTab(workspace1, "Workspace 1");
	tab_widget->addTab(workspace2, "Workspace 2");

	PVGuiQt::PVDockWidget* dock_widget1 = new PVGuiQt::PVDockWidget();
	QLabel* label1 = new QLabel("LABEL1");
	std::cout << "label1:" << label1 << std::endl;
	dock_widget1->setWidget(label1);
	dock_widget1->setWindowTitle("View 1");
	label1->setStyleSheet("background-color: purple;");

	PVGuiQt::PVDockWidget* dock_widget2 = new PVGuiQt::PVDockWidget();
	dock_widget2->setStyleSheet("background-color: blue;");
	QLabel* label2 = new QLabel();
	dock_widget2->setWidget(label2);
	dock_widget2->setWindowTitle("View 2");

	PVGuiQt::PVDockWidget* dock_widget3 = new PVGuiQt::PVDockWidget();
	dock_widget3->setStyleSheet("background-color: green;");
	QLabel* label3 = new QLabel();
	dock_widget3->setWidget(label3);
	dock_widget3->setWindowTitle("View 3");

	PVGuiQt::PVDockWidget* dock_widget4 = new PVGuiQt::PVDockWidget();
	dock_widget4->setStyleSheet("background-color: red;");
	QLabel* label4 = new QLabel();
	dock_widget4->setWidget(label4);
	dock_widget4->setWindowTitle("View 4");

	PVGuiQt::PVDockWidget* dock_widget5 = new PVGuiQt::PVDockWidget();
	dock_widget5->setStyleSheet("background-color: red;");
	QLabel* label5 = new QLabel();
	QLabel* label6 = new QLabel("Label6");
	dock_widget5->setWidget(label5);
	dock_widget5->setWidget(label6);
	dock_widget5->setWindowTitle("View 5");

	workspace1->addDockWidget(Qt::TopDockWidgetArea, dock_widget1);
	workspace1->addDockWidget(Qt::TopDockWidgetArea, dock_widget2);
	workspace1->tabifyDockWidget(dock_widget1, dock_widget2);
	workspace1->addDockWidget(Qt::TopDockWidgetArea, dock_widget3);
	workspace1->tabifyDockWidget(dock_widget2, dock_widget3);
	workspace1->addDockWidget(Qt::TopDockWidgetArea, dock_widget4);
	workspace1->tabifyDockWidget(dock_widget3, dock_widget4);

	dock_widget1->raise(); // Set active the specified PVGuiQt::PVDockWidget

	QPushButton* button = new QPushButton("set view1 to central widget");
	workspace1->setCentralWidget(button);

	PVGuiQt::SlotHandler slot_handler(workspace1, dock_widget1);
	QObject::connect(button, SIGNAL(clicked()), &slot_handler, SLOT(switch_displays()));

	workspace2->addDockWidget(Qt::TopDockWidgetArea, dock_widget5);

	mw->show();

	return app.exec();
}
