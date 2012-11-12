/**
 * \file import_source_to_project_dlg.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvguiqt/PVImportSourceToProjectDlg.h>

#include <QApplication>
#include <QStringList>

int main(int argc, char** argv)
{
	// Qt app
	QApplication app(argc, argv);

	/*PVGuiQt::PVImportSourceToProjectDlg dlg(list, 2);
	dlg.exec();*/

	return app.exec();
}
