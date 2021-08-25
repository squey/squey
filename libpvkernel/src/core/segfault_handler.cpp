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
#include <sys/types.h>

#include <pvbase/general.h>

#include <client/linux/handler/exception_handler.h>

#define BREAKPAD_MINIDUMP_FOLDER "/tmp/inendi-inspector_" INENDI_CURRENT_VERSION_STR "_coredumps"

static bool dump_callback(const google_breakpad::MinidumpDescriptor& descriptor,
                          void* /*context*/,
                          bool succeeded)
{
	if (fork() == 0) {
		/* we are in the child process
		 */

		execlp("inendi-crashreport", "inendi-crashreport", descriptor.path(), nullptr);

		// if execlp returns (i.e. it has failed), we print the message in the log
		PVLOG_ERROR("Crash report file: %s\n", descriptor.path());
	}

	return succeeded;
}

void init_segfault_handler()
{
	mkdir(BREAKPAD_MINIDUMP_FOLDER, S_IRWXU | S_IRGRP | S_IXGRP);
	static google_breakpad::MinidumpDescriptor descriptor(BREAKPAD_MINIDUMP_FOLDER);
	static google_breakpad::ExceptionHandler eh(descriptor, nullptr, dump_callback, nullptr, true,
	                                            -1);
}
