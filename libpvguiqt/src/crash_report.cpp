/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2018
 */

#include <QApplication>

#include <iostream>
#include "include/pvguiqt/PVCrashReporterDialog.h"

int main(int argc, char* argv[])
{
	QApplication app(argc, argv);

	if (argc < 2) {
		std::cerr << "usage: " << argv[0] << " <minidump_path>" << std::endl;
		exit(1);
	}
	std::string minidump_path = argv[1];

	PVGuiQt::PVCrashReporterDialog crash_reporter(minidump_path);
	crash_reporter.show();

	return app.exec();
}
