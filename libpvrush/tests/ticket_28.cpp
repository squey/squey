#include <pvrush/PVFormat.h>
#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

#define FILES_DIR "test-files/tickets/28/"

int main()
{
	PVRush::PVFormat format("org", FILES_DIR "field_enum.format");
	format.populate();

	int fd = open(FILES_DIR "field_enum.format", O_RDWR);
	if (fd == -1) {
		std::cerr << "Unable to open the format for reading/writing after PVFormat::populate() : " << strerror(errno) << std::endl;
		return 1;
	}
	return 0;
}
