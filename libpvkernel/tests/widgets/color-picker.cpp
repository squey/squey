#include <pvkernel/widgets/PVColorPicker.h>
#include <QApplication>

#include <QMainWindow>

int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	PVWidgets::PVColorPicker* cp = new PVWidgets::PVColorPicker();
	cp->set_h_offset(0);
	cp->set_color(192/2);
	QMainWindow* mw = new QMainWindow();
	mw->setCentralWidget(cp);

	mw->show();

	return app.exec();
}
