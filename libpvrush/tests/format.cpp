#include <pvrush/PVFormat.h>
#include <pvrush/PVUnicodeSource.h>
#include <pvrush/PVInputFile.h>

#include <pvfilter/PVPluginsLoad.h>

#include <QCoreApplication>
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
	QCoreApplication app(argc, argv);
	// Format reading
	QDir dir_files(argv[1]);
	dir_files.setFilter(QDir::Files | QDir::Readable);
	QStringList files = dir_files.entryList(QStringList() << QString("*.format"));

	for (int i = 0; i < files.size(); i++) {
		QString fpath = dir_files.absoluteFilePath(files[i]);
		PVRush::PVFormat format("format", fpath);
		if (!format.populate()) {
			std::cerr << "Unable to populate format from file " << qPrintable(fpath) << std::endl;
			return 1;
		}

		PVFilter::PVChunkFilter_f f = format.create_tbb_filters();
		if (!f) {
			std::cerr << "Unable to create filters for format from file " << qPrintable(fpath) << std::endl;
			return 1;
		}
	}

	return 0;
}
