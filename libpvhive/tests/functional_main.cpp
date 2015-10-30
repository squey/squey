/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <QApplication>

#include "functional_dlg.h"

int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	FunctionalDlg dlg(nullptr);

	dlg.show();

	app.exec();

	return 0;
}
