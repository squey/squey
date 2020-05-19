/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

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
