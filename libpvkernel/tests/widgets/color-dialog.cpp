/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/widgets/PVColorDialog.h>
#include <QApplication>

#include <QMainWindow>

int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	PVWidgets::PVColorDialog* cp = new PVWidgets::PVColorDialog();
	cp->set_color(20);
	cp->show();

	return app.exec();
}
