//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvguiqt/PVCrashReporterDialog.h>
#include <pvkernel/core/segfault_handler.h>
#include <QApplication>
#include <QFile>
#include <QFileInfo>
#include <QString>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
	QApplication app(argc, argv);

	// Spawned by the Crashpad handler right after a crash, with the envelope as
	// its argument, or by squey at start-up when a report from a previous run is
	// still waiting. The envelope is a Sentry format the crash server does not
	// take, so it is discarded and the minidump is read from the database. An
	// explicit minidump path is still accepted, which is what makes a report
	// sendable by hand.
	std::string minidump_path;
	if (argc > 1) {
		const QString argument = QString::fromLocal8Bit(argv[1]);
		if (argument.endsWith(".dmp") and QFileInfo(argument).isFile()) {
			minidump_path = argument.toStdString();
		} else {
			QFile::remove(argument);
		}
	}
	if (minidump_path.empty()) {
		minidump_path = pending_crash_report_path();
	}

	if (minidump_path.empty()) {
		std::cerr << "usage: " << argv[0] << " [minidump_path]" << std::endl;
		std::cerr << "no crash report waiting to be sent" << std::endl;
		return 1;
	}

	PVGuiQt::PVCrashReporterDialog crash_reporter(minidump_path);
	crash_reporter.show();

	return app.exec();
}
