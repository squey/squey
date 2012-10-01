#include <pvkernel/widgets/PVColorPicker.h>
#include <QApplication>

#include <QMainWindow>

int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	PVWidgets::PVColorPicker* cp = new PVWidgets::PVColorPicker();
	cp->set_allow_empty_interval(false);
	cp->set_x0(HSV_COLOR_GREEN);
	cp->set_x1(HSV_COLOR_RED);
	//cp->set_selection_mode(PVWidgets::PVColorPicker::SelectionInterval);
	//cp->set_interval(HSV_COLOR_GREEN+4, HSV_COLOR_RED-4);
	cp->set_color(HSV_COLOR_GREEN+4);
	QMainWindow* mw = new QMainWindow();
	mw->setCentralWidget(cp);

	mw->show();

	return app.exec();
}
