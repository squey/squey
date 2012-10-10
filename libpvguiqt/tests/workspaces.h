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


class CustomDockWidget : public QDockWidget
{
	Q_OBJECT

public:
	CustomDockWidget(QWidget* parent = 0) : QDockWidget(parent) {}

protected:
	bool event(QEvent* event);
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
};


#endif /* WORKSPACES_H_ */
