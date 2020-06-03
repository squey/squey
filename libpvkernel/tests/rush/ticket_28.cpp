/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <iostream>
#include <filesystem>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

#include "common.h"

#define FILES_DIR "../../tests/files/pvkernel/run/tickets/28/"

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " test-files-directory" << std::endl;
		return 1;
	}

	pvtest::init_ctxt();

	const QString format_path =
	    QString::fromLocal8Bit(argv[1]) + QLatin1String("/tickets/28/field_enum.format");
	const std::string& out_path = pvtest::get_tmp_filename();
	std::filesystem::copy(format_path.toStdString(), out_path);
	PVRush::PVFormat format("org", QString::fromStdString(out_path));

	int fd = open(out_path.c_str(), O_RDWR);
	std::remove(out_path.c_str());
	if (fd == -1) {
		std::cerr << "Unable to open the format for reading/writing after PVFormat::populate() : "
		          << strerror(errno) << std::endl;
		return 1;
	}
	return 0;
}
