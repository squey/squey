#include <QtGui>

#include <picviz/PVAD2GView.h>
#include <picviz/widgets/PVAD2GWidget.h>
#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>


int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	QMainWindow w;

	Picviz::PVRoot_p root(new Picviz::PVRoot());
	Picviz::PVScene_p scene(new Picviz::PVScene("scene", root.get()));
	Picviz::PVAD2GView_p ad2gv(new Picviz::PVAD2GView(scene.get()));

	PVWidgets::PVAD2GWidget* ad2g_widget = new PVWidgets::PVAD2GWidget(ad2gv, &w);
	w.setCentralWidget(ad2g_widget);
	w.show();

	delete ad2g_widget;

	return a.exec();
}
