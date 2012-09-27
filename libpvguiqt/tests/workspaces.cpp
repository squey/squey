/**
 * \file workspaces.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <QApplication>
#include <QMainWindow>
#include <QTabWidget>
#include <QDockWidget>
#include <QStyle>
#include <QDesktopWidget>
#include <QLabel>
#include <QLabel>

class CustomMainWindow : public QMainWindow
{
public:

	CustomMainWindow()
	{
		setMinimumSize(500,600);

		setGeometry(
		    QStyle::alignedRect(
		        Qt::LeftToRight,
		        Qt::AlignCenter,
		        size(),
		        qApp->desktop()->availableGeometry()
		    ));
	}
};

class FakeMainWindow : public QMainWindow
{
public:

	FakeMainWindow()
	{

	}
};

int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	QMainWindow* mw = new CustomMainWindow();

	QWidget* main_widget = new QWidget();

	FakeMainWindow* fake_main_window1 = new FakeMainWindow();
	fake_main_window1->setTabPosition(Qt::TopDockWidgetArea, QTabWidget::North);

	FakeMainWindow* fake_main_window2 = new FakeMainWindow();
	fake_main_window2->setTabPosition(Qt::TopDockWidgetArea, QTabWidget::North);

	QTabWidget* tab_widget = new QTabWidget(mw);
	tab_widget->resize(640, 600);
	tab_widget->setTabsClosable(true);

	tab_widget->addTab(fake_main_window1, "Workspace 1");
	tab_widget->addTab(fake_main_window2, "Workspace 2");

	QDockWidget* dock_widget1 = new QDockWidget();
	QLabel* label1 = new QLabel();
	dock_widget1->setWidget(label1);
	dock_widget1->setWindowTitle("View 1");
	dock_widget1->setStyleSheet("background-color: purple;");
	QDockWidget* dock_widget2 = new QDockWidget();
	dock_widget2->setStyleSheet("background-color: blue;");
	QLabel* label2 = new QLabel();
	dock_widget2->setWidget(label2);
	dock_widget2->setWindowTitle("View 2");
	QDockWidget* dock_widget3 = new QDockWidget();
	dock_widget3->setStyleSheet("background-color: green;");
	QLabel* label3 = new QLabel();
	dock_widget3->setWidget(label3);
	dock_widget3->setWindowTitle("View 3");

	QDockWidget* dock_widget4 = new QDockWidget();
	dock_widget4->setWindowTitle("View 4");

	fake_main_window1->addDockWidget(Qt::TopDockWidgetArea, dock_widget1);
	fake_main_window1->addDockWidget(Qt::TopDockWidgetArea, dock_widget2);
	fake_main_window1->tabifyDockWidget(dock_widget1, dock_widget2);
	fake_main_window1->addDockWidget(Qt::TopDockWidgetArea, dock_widget3);
	fake_main_window1->tabifyDockWidget(dock_widget2, dock_widget3);

	dock_widget1->raise(); // Set active the specified QDockWidget


	fake_main_window2->addDockWidget(Qt::TopDockWidgetArea, dock_widget4);

	mw->show();

	return app.exec();
}
