/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <pvkernel/core/segfault_handler.h>
#include <pvkernel/core/PVLogger.h>

#include <unistd.h>

#include <client/linux/handler/exception_handler.h>

static bool dump_callback(const google_breakpad::MinidumpDescriptor& descriptor, void* /*context*/,
                          bool succeeded)
{
	if (fork() == 0) {
		/* we are in the child process
		 */

		// xmessage is part of x11-utils which is always installed with X11
		execlp("xmessage", "xmessage", "-title", "INENDI Inspector crash report", "-center", "The crash report file path is:", descriptor.path(), nullptr);

		// if execlp returns (i.e. it has failed), we print the message in the log
		PVLOG_ERROR("Crash report file: %s\n", descriptor.path());
	}

	return succeeded;
}

void init_segfault_handler()
{
	static google_breakpad::MinidumpDescriptor descriptor("/tmp");
	static google_breakpad::ExceptionHandler eh(descriptor, nullptr, dump_callback, nullptr, true, -1);
}
