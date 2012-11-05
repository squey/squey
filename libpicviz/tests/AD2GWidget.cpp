/**
 * \file AD2GWidget.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <QtGui>

#include <picviz/PVAD2GView.h>
#include <picviz/widgets/PVAD2GWidget.h>
#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>


int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	QMainWindow w;

	Picviz::PVRoot_sp root = Picviz::PVRoot::get_root_sp();
	Picviz::PVScene_p scene(root, "scene");
	Picviz::PVAD2GView_p ad2gv(new Picviz::PVAD2GView());

	PVWidgets::PVAD2GWidget* ad2g_widget = new PVWidgets::PVAD2GWidget(ad2gv, &w);
	w.setCentralWidget(ad2g_widget);
	w.show();

	delete ad2g_widget;

	return a.exec();
}
