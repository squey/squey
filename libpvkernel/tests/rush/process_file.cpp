#define SIMULATE_PIPELINE
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <cstdlib>
#include <iostream>
#include "helpers.h"
#include <QCoreApplication>
#include "test-env.h"

using std::cout;
using std::cerr;
using std::endl;

int main(int argc, char** argv)
{
	if (argc <= 2) {
		cerr << "Usage: " << argv[0] << " file format" << endl;
		return 1;
	}

	init_env();
	QCoreApplication app(argc, argv);
	PVFilter::PVPluginsLoad::load_all_plugins();
	PVRush::PVPluginsLoad::load_all_plugins();

	// Input file
	QString path_file(argv[1]);
	PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(path_file));

	// Load the given format file
	QString path_format(argv[2]);
	PVRush::PVFormat format("format", path_format);
	if (!format.populate(true)) {
		std::cerr << "Can't read format file " << qPrintable(path_format) << std::endl;
		return 1;
	}

	// Get the source creator
	QString file_path(argv[1]);
	PVRush::PVSourceCreator_p sc_file;
	if (!PVRush::PVTests::get_file_sc(file, format, sc_file)) {
		return 1;
	}

	// Process that file with the found source creator thanks to the extractor
	PVRush::PVSourceCreator::source_p src = sc_file->create_source_from_input(file, format);
	if (!src) {
		std::cerr << "Unable to create PVRush source from file " << argv[1] << std::endl;
		return 1;
	}

	// Create the extractor
	PVRush::PVExtractor ext;
	ext.start_controller();
	ext.add_source(src);
	ext.set_chunk_filter(format.create_tbb_filters());

	// Ask for 1 million lines
	PVRush::PVControllerJob_p job = ext.process_from_agg_nlines(0, 1000000);
	job->wait_end();

	// Dump the NRAW to stdout
	dump_nraw_csv(ext.get_nraw());

	return 0;
}
