
#include <pvkernel/core/PVFileHelper.h>

#include <pvkernel/core/PVLogger.h>

#ifdef WIN32
#error PVCore::PVFileHelper::is_already_opened is not implemented for MS Windows environments
#else
#include <unistd.h>
#include <fcntl.h>
#endif

bool PVCore::PVFileHelper::is_already_opened(const char* file_name)
{
	bool ret = false;
#ifdef WIN32
#else
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

#endif
    return ret;
}
