/**
 * \file ticket_28.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

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

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " test-files-directory" << std::endl;
		return 1;
	}

	init_env();
	PVFilter::PVPluginsLoad::load_all_plugins();
	PVRush::PVPluginsLoad::load_all_plugins();

	const QString format_path = QString::fromLocal8Bit(argv[1]) + QLatin1String("/tickets/28/field_enum.format");
	PVRush::PVFormat format("org", format_path);
	format.populate();

	int fd = open(qPrintable(format_path), O_RDWR);
	if (fd == -1) {
		std::cerr << "Unable to open the format for reading/writing after PVFormat::populate() : " << strerror(errno) << std::endl;
		return 1;
	}
	return 0;
}
