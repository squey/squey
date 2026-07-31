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

/**
 * Checks that declined crash reports are dropped from the database, so that they
 * neither pile up nor keep being offered.
 *
 * Only the declining paths are exercised: accepting would post the report to the
 * crash server, which a test has no business doing.
 */

#include <pvguiqt/PVCrashReporterDialog.h>
#include <pvkernel/core/segfault_handler.h>
#include <pvkernel/core/squey_assert.h>

#include <client/crash_report_database.h>
#include <base/files/file_path.h>

#include <QApplication>
#include <QDir>
#include <QString>

#include <iostream>
#include <memory>
#include <string>

namespace
{

base::FilePath to_file_path(const std::string& path)
{
#ifdef _WIN32
	return base::FilePath(std::wstring(path.begin(), path.end()));
#else
	return base::FilePath(path);
#endif
}

/* A report the handler would have written, reduced to the minidump signature:
 * nothing here reads the dump itself.
 */
void write_report(crashpad::CrashReportDatabase& database)
{
	std::unique_ptr<crashpad::CrashReportDatabase::NewReport> new_report;
	PV_VALID(int(database.PrepareNewCrashReport(&new_report)),
	         int(crashpad::CrashReportDatabase::kNoError));

	static constexpr char MINIDUMP_SIGNATURE[] = "MDMP";
	PV_ASSERT_VALID(new_report->Writer()->Write(MINIDUMP_SIGNATURE,
	                                            sizeof(MINIDUMP_SIGNATURE) - 1));

	crashpad::UUID uuid;
	PV_VALID(int(database.FinishedWritingCrashReport(std::move(new_report), &uuid)),
	         int(crashpad::CrashReportDatabase::kNoError));
}

/* Declining through the dialog, the way closing the window or pressing Escape
 * does: both end up in QDialog::reject().
 */
void decline_through_dialog(const std::string& minidump_path)
{
	PVGuiQt::PVCrashReporterDialog dialog(minidump_path);
	dialog.reject();
}

} // namespace

int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	// SQUEY_CONFIG_DIR is set by CTest, and has to be: PVConfig reads it when the
	// library is loaded, so setting it here would come too late. It is relative,
	// so that it resolves wherever the testsuite is unpacked.

	const std::string database_path = crash_report_database_path();
	QDir(QString::fromStdString(database_path)).removeRecursively();
	PV_ASSERT_VALID(QDir().mkpath(QString::fromStdString(database_path)), "database",
	                database_path);

	std::unique_ptr<crashpad::CrashReportDatabase> database =
	    crashpad::CrashReportDatabase::Initialize(to_file_path(database_path));
	PV_ASSERT_VALID(database != nullptr, "database", database_path);

	// A declined report is dropped, and is not offered again.
	write_report(*database);
	const std::string report_path = pending_crash_report_path();
	PV_ASSERT_VALID(not report_path.empty());
	decline_through_dialog(report_path);
	PV_ASSERT_VALID(pending_crash_report_path().empty(), "left over", pending_crash_report_path());

	// Several reports pile up while squey is not running. Only the most recent
	// one is ever offered, so each has to be dropped as it is declined,
	// otherwise the older ones stay in the database for good.
	static constexpr int REPORT_COUNT = 3;
	for (int i = 0; i < REPORT_COUNT; ++i) {
		write_report(*database);
	}

	int declined = 0;
	for (std::string path = pending_crash_report_path(); not path.empty();
	     path = pending_crash_report_path()) {
		decline_through_dialog(path);
		++declined;
		// A report that is not dropped would hand out the same path for ever.
		PV_ASSERT_VALID(declined <= REPORT_COUNT, "declined", declined);
	}
	PV_VALID(declined, REPORT_COUNT);

	std::cout << declined << " declined report(s) dropped from the database" << std::endl;

	return 0;
}
