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
#include <unistd.h>
#include <sys/stat.h>
#include <client/linux/handler/exception_handler.h>
#include <client/linux/handler/minidump_descriptor.h>
#include <fcntl.h>
#include <limits.h>
#include <memory>
#include <array>
#include <string>

#include <boost/dll/runtime_symbol_info.hpp>

#include <pvbase/general.h> // IWYU pragma: keep

#define BREAKPAD_MINIDUMP_FOLDER "/tmp/squey_" SQUEY_CURRENT_VERSION_STR "_coredumps"

// Absolute path of the crash reporter, resolved once at startup. execlp() used to
// look the binary up in PATH, which any writable entry there could hijack, and
// building the path from the handler would allocate, which is not signal safe.
static std::array<char, PATH_MAX> g_crash_reporter_path{};

static bool dump_callback(const google_breakpad::MinidumpDescriptor& descriptor,
                          void* /*context*/,
                          bool succeeded)
{
	if (g_crash_reporter_path[0] != '\0' and fork() == 0) {
		/* we are in the child process
		 */

		// Absolute path resolved at startup, no shell and no PATH lookup
		// nosemgrep
		execl(g_crash_reporter_path.data(), g_crash_reporter_path.data(), descriptor.path(),
		      nullptr);

		// if execl returns (i.e. it has failed), we print the message in the log
		PVLOG_ERROR("Crash report file: %s\n", descriptor.path());
	}

	return succeeded;
}

void init_segfault_handler()
{
	const std::string reporter_path =
	    boost::dll::program_location().parent_path().string() + "/squey-crashreport";
	if (reporter_path.size() < g_crash_reporter_path.size()) {
		reporter_path.copy(g_crash_reporter_path.data(), reporter_path.size());
	} else {
		PVLOG_ERROR("Crash reporter path is too long: %s\n", reporter_path.c_str());
	}

	mkdir(BREAKPAD_MINIDUMP_FOLDER, S_IRWXU | S_IRGRP | S_IXGRP);
	static google_breakpad::MinidumpDescriptor descriptor(BREAKPAD_MINIDUMP_FOLDER);
	static google_breakpad::ExceptionHandler eh(descriptor, nullptr, dump_callback, nullptr, true,
	                                            -1);
}
