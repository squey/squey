/**
 * \file workspaces.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef WORKSPACES_H_
#define WORKSPACES_H_

#include <QApplication>
#include <QMainWindow>
#include <QTabWidget>
#include <QStyle>
#include <QDesktopWidget>
#include <QLabel>
#include <QPushButton>
#include <QDockWidget>
#include <QDragEnterEvent>
#include <QDateTime>
#include <iostream>

class CustomDockWidget : public QDockWidget
{
	Q_OBJECT

public:
	CustomDockWidget(QWidget* parent = 0) : QDockWidget(parent) {}

protected:
	bool event(QEvent* event);

	QPoint _press_pt;
};



class CustomMainWindow : public QMainWindow
{
	Q_OBJECT

public:

	CustomMainWindow();

public:
	void CreateDockWidgets();

public slots:
	void dragStarted(bool started);

protected:
	bool event(QEvent* event) override
	{
		//std::cout << QDateTime::currentDateTime().toMSecsSinceEpoch() << "QMainWindow receive event type: " << event->type() << std::endl;
		return QMainWindow::event(event);
	}
};

class MyEventFilter: public QObject
{
	Q_OBJECT

protected:
	bool eventFilter(QObject *obj, QEvent *ev) override;
};


#endif /* WORKSPACES_H_ */
