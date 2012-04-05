#include <QtGui>

#include <picviz/PVAD2GView.h>
#include <picviz/widgets/PVAD2GWidget.h>


int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	QMainWindow w;

	Picviz::PVAD2GView view;
	Picviz::PVAD2GWidget* ad2g_widget = new Picviz::PVAD2GWidget(view, &w);
	w.setCentralWidget(ad2g_widget->get_widget());
	w.show();

	return a.exec();
}
