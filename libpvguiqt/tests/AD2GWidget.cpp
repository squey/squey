/**
 * \file AD2GWidget.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <QApplication>
#include <QMainWindow>

#include <pvguiqt/PVAD2GWidget.h>

#include <picviz/PVAD2GView.h>
#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>


int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	QMainWindow w;

	Picviz::PVRoot_p root;
	Picviz::PVScene_p scene(root, "scene");
	Picviz::PVAD2GView_p ad2gv(new Picviz::PVAD2GView("test"));

	PVGuiQt::PVAD2GWidget* ad2g_widget = new PVGuiQt::PVAD2GWidget(ad2gv, *root, &w);
	w.setCentralWidget(ad2g_widget);
	w.show();

	delete ad2g_widget;

	return a.exec();
}
