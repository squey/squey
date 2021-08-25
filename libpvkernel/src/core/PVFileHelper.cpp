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

#include <pvkernel/core/PVFileHelper.h>
#include <pvkernel/core/PVLogger.h>

#include <cerrno> // for EAGAIN, errno
#include <fcntl.h>
#include <unistd.h>

bool PVCore::PVFileHelper::is_already_opened(const char* file_name)
{
	errno = 0;
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
