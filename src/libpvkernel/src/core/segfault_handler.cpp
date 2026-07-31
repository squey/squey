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

#include <pvkernel/core/PVLogger.h>
#include <pvkernel/core/segfault_handler.h>
#include <pvkernel/core/PVConfig.h>

#include <client/crash_report_database.h>
#include <client/crashpad_client.h>
#include <client/crashpad_info.h>
#include <client/settings.h>
#include <base/files/file_path.h>
#ifdef _WIN32
#include <base/strings/utf_string_conversions.h>
#endif

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <boost/dll/runtime_symbol_info.hpp>

#include <QDir>
#include <QProcess>
#include <QString>

#include <pvbase/general.h> // IWYU pragma: keep

namespace
{

#ifdef _WIN32
constexpr const char* EXE_SUFFIX = ".exe";
#else
constexpr const char* EXE_SUFFIX = "";
#endif

/* Crashpad paths are wide strings on Windows and narrow ones elsewhere. The
 * conversion has to go through UTF-8 rather than widening each byte, otherwise
 * a user directory holding non-ASCII characters is mangled.
 */
base::FilePath to_file_path(const std::string& path)
{
#ifdef _WIN32
	return base::FilePath(base::UTF8ToWide(path));
#else
	return base::FilePath(path);
#endif
}

std::string from_file_path(const base::FilePath& path)
{
#ifdef _WIN32
	return base::WideToUTF8(path.value());
#else
	return path.value();
#endif
}

} // namespace

std::string crash_report_database_path()
{
	return QDir::cleanPath(PVCore::PVConfig::user_dir() + SQUEY_CRASH_REPORTS_DIRNAME)
	    .toStdString();
}

namespace
{

/* Reports that have not reached the crash server yet. With uploads disabled the
 * handler does not leave every report pending: on macOS it marks them as
 * skipped right away, which files them under the completed ones. Both lists are
 * therefore walked, keeping only what was never uploaded.
 */
std::vector<crashpad::CrashReportDatabase::Report>
unsent_reports(crashpad::CrashReportDatabase& database)
{
	std::vector<crashpad::CrashReportDatabase::Report> reports;
	if (database.GetPendingReports(&reports) != crashpad::CrashReportDatabase::kNoError) {
		reports.clear();
	}

	std::vector<crashpad::CrashReportDatabase::Report> completed;
	if (database.GetCompletedReports(&completed) == crashpad::CrashReportDatabase::kNoError) {
		for (const crashpad::CrashReportDatabase::Report& report : completed) {
			if (not report.uploaded) {
				reports.push_back(report);
			}
		}
	}

	return reports;
}

/* Opening a database that is not there is the ordinary case before the first
 * crash, but Crashpad logs the failed stat() as an error. The directory is
 * therefore checked first, so that a run without any report stays quiet.
 */
std::unique_ptr<crashpad::CrashReportDatabase> open_existing_database()
{
	const std::string database_path = crash_report_database_path();
	if (not QDir(QString::fromStdString(database_path)).exists()) {
		return nullptr;
	}

	return crashpad::CrashReportDatabase::InitializeWithoutCreating(
	    to_file_path(database_path));
}

} // namespace

std::string pending_crash_report_path()
{
	std::unique_ptr<crashpad::CrashReportDatabase> database = open_existing_database();
	if (not database) {
		return {};
	}

	const crashpad::CrashReportDatabase::Report* most_recent = nullptr;
	const std::vector<crashpad::CrashReportDatabase::Report> reports = unsent_reports(*database);
	for (const crashpad::CrashReportDatabase::Report& report : reports) {
		if (most_recent == nullptr or report.creation_time > most_recent->creation_time) {
			most_recent = &report;
		}
	}

	return most_recent != nullptr ? from_file_path(most_recent->file_path) : std::string();
}

void discard_crash_report(const std::string& minidump_path)
{
	std::unique_ptr<crashpad::CrashReportDatabase> database = open_existing_database();
	if (not database) {
		return;
	}

	const std::vector<crashpad::CrashReportDatabase::Report> reports = unsent_reports(*database);
	for (const crashpad::CrashReportDatabase::Report& report : reports) {
		if (from_file_path(report.file_path) == minidump_path) {
			database->DeleteReport(report.uuid);
			return;
		}
	}
}

void init_segfault_handler()
{
	const std::string exe_dir = boost::dll::program_location().parent_path().string();
	const std::string handler_path = exe_dir + "/crashpad_handler" + EXE_SUFFIX;
	const std::string reporter_path = exe_dir + "/squey-crashreport" + EXE_SUFFIX;

	const std::string database_path = crash_report_database_path();
	QDir().mkpath(QString::fromStdString(database_path));

	// The database outlives the handler process, so a report that could not be
	// sent right after the crash is still offered on the next start-up.
	std::unique_ptr<crashpad::CrashReportDatabase> database =
	    crashpad::CrashReportDatabase::Initialize(to_file_path(database_path));
	if (not database or not database->GetSettings()) {
		PVLOG_ERROR("Could not initialize the crash report database in '%s'\n",
		            database_path.c_str());
		return;
	}

	// Crashpad never uploads on its own: squey-crashreport asks for consent
	// first and posts the minidump itself.
	database->GetSettings()->SetUploadsEnabled(false);

	// Left alone, Crashpad also hands the exception over to the crash reporter of
	// the system, which on macOS puts its own "Squey quit unexpectedly" window
	// next to ours. One report and one dialog are enough.
	crashpad::CrashpadInfo::GetCrashpadInfo()->set_system_crash_reporter_forwarding(
	    crashpad::TriState::kDisabled);

	// Annotations end up as fields of the crash report, which is what lets the
	// server group reports per product and version.
	const std::map<std::string, std::string> annotations = {
	    {"product", SQUEY_CRASH_REPORT_PRODUCT},
	    {"version", SQUEY_CURRENT_VERSION_STR},
	};

	const std::vector<std::string> arguments = {"--no-rate-limit"};

	static crashpad::CrashpadClient client;
	// squey-crashreport is spawned by the handler as soon as the minidump has
	// been written, so consent is asked for while the crash is still fresh. It
	// reads the report from the database, which the keep-report-for-crash-reporter
	// patch preserves: upstream deletes it, assuming the reporter reads the
	// minidump back from the Sentry envelope. The envelope is written all the
	// same, since the handler only spawns the reporter when it has one, and
	// squey-crashreport discards it.
	const bool started = client.StartHandler(
	    to_file_path(handler_path), to_file_path(database_path), to_file_path(database_path),
	    /* url */ std::string(), /* http_proxy */ std::string(), annotations, arguments,
	    /* restartable */ true, /* asynchronous_start */ false, /* attachments */ {},
	    /* screenshot */ base::FilePath(), /* wait_for_upload */ false,
	    to_file_path(reporter_path),
	    to_file_path(database_path + "/" + SQUEY_CRASH_ENVELOPE_FILENAME));

	if (not started) {
		PVLOG_ERROR("Could not start the crash handler '%s'\n", handler_path.c_str());
	}

	// A report left over from a previous run: ask the user whether to send it.
	// Detached, so that a crash reporter still waiting for an answer does not
	// hold squey back. The database opened above is reused rather than opened a
	// second time, which would contend on its lock files.
	if (not unsent_reports(*database).empty()) {
		QProcess::startDetached(QString::fromStdString(reporter_path), {});
	}
}
