/**
 * \file correlation_menu.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <QApplication>
#include <QMainWindow>
#include <QMenuBar>
#include <QMenu>
#include <QAction>

#include <pvguiqt/PVCorrelationMenu.h>

int main(int argc, char** argv)
{
	// Qt app
	QApplication app(argc, argv);

	QMainWindow* mw = new QMainWindow();

	mw->show();

	QMenuBar* menu_bar = new QMenuBar();
	mw->setMenuBar(menu_bar);

	menu_bar->addMenu(new PVGuiQt::PVCorrelationMenu());

	//connect(action_create_correlation, SIGNAL(triggered(bool)), this, SLOT(activer_enregistrement(bool)));

	return app.exec();
}
