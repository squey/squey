/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVFileHelper.h>
#include <pvkernel/core/PVLogger.h>

#include <cerrno> // for EAGAIN, errno
#include <fcntl.h>
#include <unistd.h>

bool PVCore::PVFileHelper::is_already_opened(const char* file_name)
{
	bool ret = false;
	int fd = open(file_name, O_RDONLY);
	if (fd < 0) {
		PVLOG_WARN("trying to test a file which does not exist: %s\n", file_name);
	} else {
		if ((fcntl(fd, F_SETLEASE, F_WRLCK) < 0) && (errno == EAGAIN)) {
			ret = true;
		} else {
			// don't forget to remove the lease
			fcntl(fd, F_SETLEASE, F_UNLCK);
		}
		close(fd);
	}

	return ret;
}
