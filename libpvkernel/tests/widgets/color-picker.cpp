/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/widgets/PVColorPicker.h>
#include <QApplication>

#include <QMainWindow>

int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	PVWidgets::PVColorPicker* cp = new PVWidgets::PVColorPicker();
	cp->set_allow_empty_interval(false);
	cp->set_x0(HSV_COLOR_GREEN.h());
	cp->set_x1(HSV_COLOR_RED.h());
	cp->set_color(PVCore::PVHSVColor(HSV_COLOR_GREEN.h() + 4));
	QMainWindow* mw = new QMainWindow();
	mw->setCentralWidget(cp);

	mw->show();

	return app.exec();
}
