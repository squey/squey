#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVUnicodeSource.h>
#include <pvkernel/rush/PVInputFile.h>

#include <pvkernel/filter/PVPluginsLoad.h>

#include <QCoreApplication>
#include <QDir>
#include <QStringList>

#include <iostream>

#include "helpers.h"
#include "test-env.h"

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " format" << std::endl;
		return 1;
	}

	// Initialisation
	init_env();
	PVFilter::PVPluginsLoad::load_all_plugins();
	QCoreApplication app(argc, argv);
	// Format reading

	QString fpath = argv[1];
	PVRush::PVFormat format("format", fpath);
	if (!format.populate(true)) {
		std::cerr << "Unable to populate format from file " << qPrintable(fpath) << std::endl;
		return 1;
	}

	format.debug();

	return 0;
}
