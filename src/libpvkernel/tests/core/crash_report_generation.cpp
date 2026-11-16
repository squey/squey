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
 * Checks that a crash is actually turned into a report, on the three platforms.
 *
 * The test re-executes itself with --crash-child: the child starts the Crashpad
 * handler and dereferences a null pointer, then the parent waits for the report
 * to show up in the database and drops it again.
 *
 * Under Linux, the child first puts in front of Crashpad, for SIGSEGV and SIGFPE,
 * a signal handler that raises the signal again from where it is, as those of
 * LLVM do, then calls reclaim_crash_signals(). The child checks that SIGFPE
 * stays with that handler, the parent that the report still places the crash
 * where it happened.
 *
 * The child needs crashpad_handler next to the test executable, which the
 * CMakeLists copies there. squey-crashreport is deliberately *not* copied: the
 * handler would spawn it and its dialog would hang the test. Failing to spawn it
 * is harmless, the minidump is written before that.
 */

#include <pvkernel/core/segfault_handler.h>
#include <pvkernel/core/squey_assert.h>

#include <QByteArray>
#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QIODevice>
#include <QProcess>
#include <QString>
#include <QStringList>
#include <QThread>
#include <QtEndian>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif
#ifdef __linux__
#include <signal.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

// The handler writes the minidump after the crashed process is gone, so the
// report only shows up shortly afterwards.
static constexpr int REPORT_WAIT_MS = 60000;
static constexpr int POLL_INTERVAL_MS = 100;

// Reported to CTest as a skip rather than a failure, see SKIP_RETURN_CODE.
static constexpr int SKIPPED_EXIT_CODE = 77;

static const std::string CRASH_SITE_PREFIX = "crash site: ";

// In a function of its own, whose address the child prints for the parent. The pointer is
// volatile too: GCC, seeing it null, would otherwise drop the call to this function.
[[gnu::noinline]] static void write_through_null()
{
	volatile int* volatile invalid_address = nullptr;
	*invalid_address = 42;
}

#ifdef __linux__
static struct sigaction displaced_actions[NSIG];

/* As the signal handlers of LLVM since version 21: the handler displaced is put
 * back, and the signal raised again with the same siginfo, from here.
 */
static void reraise_from_here(int sig, siginfo_t* info, void*)
{
	sigaction(sig, &displaced_actions[sig], nullptr);
	syscall(SYS_rt_tgsigqueueinfo, getpid(), gettid(), sig, info);
}

static void put_reraising_handler_in_front(int sig)
{
	struct sigaction action = {};
	action.sa_sigaction = reraise_from_here;
	// LLVM unblocks every signal instead: either way, the signal raised again is
	// delivered at once, before the handler returns.
	action.sa_flags = SA_SIGINFO | SA_NODEFER;
	sigemptyset(&action.sa_mask);
	sigaction(sig, &action, &displaced_actions[sig]);
}

static bool handled_by_reraise_from_here(int sig)
{
	struct sigaction action = {};
	return sigaction(sig, nullptr, &action) == 0 and action.sa_sigaction == reraise_from_here;
}
#endif

static int crash_on_purpose()
{
#ifdef _WIN32
	// Otherwise Windows pops up an error dialog and the test hangs until the
	// CTest timeout instead of crashing.
	SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#endif

	init_segfault_handler();

#ifdef __linux__
	put_reraising_handler_in_front(SIGSEGV);
	put_reraising_handler_in_front(SIGFPE);
	reclaim_crash_signals();
	if (not handled_by_reraise_from_here(SIGFPE)) {
		std::cout << "reclaim_crash_signals() took SIGFPE, which PortableCL needs first"
		          << std::endl;
		return 1;
	}
#endif

	// Printed so that a mismatch with the database of the parent is visible.
	std::cout << "child database: " << crash_report_database_path() << std::endl;
	std::cout << CRASH_SITE_PREFIX << std::hex
	          << reinterpret_cast<std::uintptr_t>(&write_through_null) << std::dec << std::endl;
	std::cout.flush();

	write_through_null();

	// Not reached: reaching it means the crash did not happen, which the parent
	// reports as a failure through the exit status.
	return 1;
}

#if defined(__linux__) and defined(__x86_64__)
template <typename T>
static std::optional<T> read_little_endian(const QByteArray& data, qsizetype offset)
{
	if (offset < 0 or offset + qsizetype(sizeof(T)) > data.size()) {
		return std::nullopt;
	}
	return qFromLittleEndian<T>(data.constData() + offset);
}

/* The instruction pointer of the context in the exception stream of a minidump:
 * where the report places the crash. The structures are those of the minidump
 * format of Breakpad, minidump_format.h.
 */
static std::optional<std::uint64_t> crash_instruction_pointer(const QByteArray& minidump)
{
	constexpr qsizetype STREAM_COUNT = 8;          // in MDRawHeader
	constexpr qsizetype STREAM_DIRECTORY_RVA = 12; // in MDRawHeader
	constexpr qsizetype DIRECTORY_ENTRY_SIZE = 12; // MDRawDirectory
	constexpr qsizetype STREAM_RVA = 8;            // in MDRawDirectory
	constexpr quint32 EXCEPTION_STREAM = 6;        // MD_EXCEPTION_STREAM
	constexpr qsizetype THREAD_CONTEXT_RVA = 164;  // in MDRawExceptionStream
	constexpr qsizetype RIP = 0xf8;                // in MDRawContextAMD64

	const std::optional<quint32> stream_count = read_little_endian<quint32>(minidump, STREAM_COUNT);
	const std::optional<quint32> directory =
	    read_little_endian<quint32>(minidump, STREAM_DIRECTORY_RVA);
	if (not stream_count or not directory) {
		return std::nullopt;
	}

	for (quint32 i = 0; i < *stream_count; ++i) {
		const qsizetype entry = *directory + i * DIRECTORY_ENTRY_SIZE;
		if (read_little_endian<quint32>(minidump, entry) != EXCEPTION_STREAM) {
			continue;
		}
		const std::optional<quint32> stream =
		    read_little_endian<quint32>(minidump, entry + STREAM_RVA);
		if (not stream) {
			return std::nullopt;
		}
		const std::optional<quint32> context =
		    read_little_endian<quint32>(minidump, *stream + THREAD_CONTEXT_RVA);
		if (not context) {
			return std::nullopt;
		}
		return read_little_endian<quint64>(minidump, *context + RIP);
	}

	return std::nullopt;
}

static std::optional<std::uint64_t> crash_site(const std::string& child_output)
{
	const size_t position = child_output.find(CRASH_SITE_PREFIX);
	if (position == std::string::npos) {
		return std::nullopt;
	}
	return std::stoull(child_output.substr(position + CRASH_SITE_PREFIX.size()), nullptr, 16);
}

static std::string hex(std::uint64_t value)
{
	return "0x" + QString::number(value, 16).toStdString();
}
#endif

/* The ptrace policy as the sandbox sees it, which is not necessarily the one of
 * the host: reported alongside a skip so that it does not have to be guessed.
 */
static std::string yama_ptrace_scope()
{
	QFile scope("/proc/sys/kernel/yama/ptrace_scope");
	if (not scope.open(QIODevice::ReadOnly)) {
		return "unreadable";
	}
	return scope.readAll().trimmed().toStdString();
}

static std::string wait_for_crash_report()
{
	QElapsedTimer timer;
	timer.start();

	while (timer.elapsed() < REPORT_WAIT_MS) {
		const std::string report_path = pending_crash_report_path();
		if (not report_path.empty()) {
			return report_path;
		}
		QThread::msleep(POLL_INTERVAL_MS);
	}

	return {};
}

int main(int argc, char* argv[])
{
	if (argc > 1 and std::strcmp(argv[1], "--crash-child") == 0) {
		return crash_on_purpose();
	}

	QCoreApplication app(argc, argv);

	// SQUEY_CONFIG_DIR is set by CTest, and has to be: PVConfig reads it when the
	// library is loaded, so setting it here would come too late. It is relative,
	// so that it resolves wherever the testsuite is unpacked -- on Windows and
	// macOS the tests run on another machine than the one that built them.

	const QString database_path = QString::fromStdString(crash_report_database_path());
	QDir(database_path).removeRecursively();

	std::cout << "crash report database: " << database_path.toStdString() << std::endl;
	PV_ASSERT_VALID(pending_crash_report_path().empty());

	QProcess child;
	// Whatever the child logs is the only clue as to why no report was written,
	// the handler being started there.
	child.setProcessChannelMode(QProcess::MergedChannels);
	child.start(QCoreApplication::applicationFilePath(), QStringList{"--crash-child"});
	PV_ASSERT_VALID(child.waitForStarted(REPORT_WAIT_MS));
	PV_ASSERT_VALID(child.waitForFinished(REPORT_WAIT_MS));

	const std::string child_output = child.readAll().toStdString();
	std::cout << "--- child output ---\n" << child_output << "--------------------" << std::endl;

	// Capturing a crash on Linux needs ptrace, which a sandbox may forbid: with
	// /proc/sys/kernel/yama/ptrace_scope at 2 or more and no CAP_SYS_PTRACE,
	// Crashpad gives up rather than falling back on its broker, and no report can
	// ever be written. That is the case of the build containers of the CI, but
	// not of squey itself, whose flatpak asks for the 'devel' feature. Crashpad
	// says so itself, which spares us from second-guessing its own detection.
	if (child_output.find("no ptrace") != std::string::npos) {
		std::cout << "skipped: this sandbox forbids ptrace, Crashpad cannot capture a crash"
		          << " (yama ptrace_scope=" << yama_ptrace_scope() << ", 2 and above need"
		          << " CAP_SYS_PTRACE)" << std::endl;
		return SKIPPED_EXIT_CODE;
	}

	// The child must have died from the crash, not returned on its own.
	PV_VALID(int(child.exitStatus()), int(QProcess::CrashExit));

	const std::string report_path = wait_for_crash_report();
	std::cout << "crash report: '" << report_path << "'" << std::endl;
	PV_ASSERT_VALID(not report_path.empty(), "child output", child_output);

	const QFileInfo report(QString::fromStdString(report_path));
	PV_ASSERT_VALID(report.isFile(), "report", report_path);
	PV_ASSERT_VALID(report.size() > 0, "report", report_path, "size", report.size());

	// A non-empty file is not enough: check it really is a minidump, so that a
	// truncated or half-written report is not taken for a valid one.
	QFile report_file(QString::fromStdString(report_path));
	PV_ASSERT_VALID(report_file.open(QIODevice::ReadOnly), "report", report_path);
	const QByteArray magic = report_file.read(4);
	PV_VALID(magic.toStdString(), std::string("MDMP"));

#if defined(__linux__) and defined(__x86_64__)
	// write_through_null() is a few instructions long. Left in front of Crashpad,
	// reraise_from_here() would have the report place the crash in libc, where
	// syscall() raised the signal again.
	report_file.seek(0);
	const std::optional<std::uint64_t> rip = crash_instruction_pointer(report_file.readAll());
	const std::optional<std::uint64_t> site = crash_site(child_output);
	PV_ASSERT_VALID(rip.has_value() and site.has_value(), "child output", child_output);
	PV_ASSERT_VALID(*rip >= *site and *rip - *site < 256, "rip", hex(*rip), "crash site",
	                hex(*site));
#endif

	// Sending or declining a report drops it, so it is not offered again.
	discard_crash_report(report_path);
	PV_ASSERT_VALID(pending_crash_report_path().empty());

	return 0;
}
