#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

#include "test-env.h"

#define FILES_DIR "test-files/tickets/28/"

int main()
{
	init_env();
	PVFilter::PVPluginsLoad::load_all_plugins();
	PVRush::PVPluginsLoad::load_all_plugins();

	PVRush::PVFormat format("org", FILES_DIR "field_enum.format");
	format.populate();

	int fd = open(FILES_DIR "field_enum.format", O_RDWR);
	if (fd == -1) {
		std::cerr << "Unable to open the format for reading/writing after PVFormat::populate() : " << strerror(errno) << std::endl;
		return 1;
	}
	return 0;
}
