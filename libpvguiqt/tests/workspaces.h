#if 0
#else

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

class CustomMainWindow;

class CustomDockWidget : public QDockWidget
{
        Q_OBJECT

public:
        CustomDockWidget(QWidget* parent = 0) : QDockWidget(parent) {}

        CustomMainWindow* workspace_under_mouse();

protected:
        bool event(QEvent* event);

        QPoint _press_pt;
};



class CustomMainWindow : public QMainWindow
{
        Q_OBJECT

public:

        CustomMainWindow(QWidget* parent = 0);

public:
        void CreateDockWidgets();

public slots:
        void dragStarted(bool started);
        void dragEnded();
        void changeEvent(QEvent *event);

public:
        int z_order() { return zOrderIndex; }

private:
        int zOrderIndex;
        static unsigned int zOrderCounter;
};


#endif /* WORKSPACES_H_ */

#endif
