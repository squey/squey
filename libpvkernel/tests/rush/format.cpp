/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVUnicodeSource.h>
#include <pvkernel/rush/PVInputFile.h>

#include <pvkernel/filter/PVPluginsLoad.h>

#include <QDir>
#include <QStringList>

#include <iostream>

#include "helpers.h"
#include "test-env.h"

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " format_dir" << std::endl;
	}

	// Initialisation
	init_env();
	PVFilter::PVPluginsLoad::load_all_plugins();
	// Format reading
	QDir dir_files(argv[1]);
	dir_files.setFilter(QDir::Files | QDir::Readable);
	QStringList files = dir_files.entryList(QStringList() << QString("*.format"));

	for (int i = 0; i < files.size(); i++) {
		QString fpath = dir_files.absoluteFilePath(files[i]);
		PVRush::PVFormat format("format", fpath);
		format.create_tbb_filters();
	}

	return 0;
}
