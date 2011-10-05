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
	init_env();
	QCoreApplication app(argc, argv);
	PVFilter::PVPluginsLoad::load_all_plugins();
	PVRush::PVPluginsLoad::load_all_plugins();

	// Input file
	QString path_file("test-files/tickets/2/apache.access");
	PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(path_file));

	// Load the given format file
	QString path_format("test-files/tickets/2/apache.access.format");
	PVRush::PVFormat format("format", path_format);
	if (!format.populate()) {
		std::cerr << "Can't read format file " << qPrintable(path_format) << std::endl;
		return false;
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

	std::cout << "Save the nraw..." << std::endl;
	// Save the NRAW
	ext.save_nraw();

	job = ext.process_from_agg_nlines(0, 10000);
	job->wait_end();

	std::cout << "Clear that save..." << std::endl;
	// Clear that save
	ext.clear_saved_nraw();

	std::cout << "Save it again..." << std::endl;
	// Save the nraw and restore it
	ext.save_nraw();

	job = ext.process_from_agg_nlines(0, 10000);
	job->wait_end();

	std::cout << "And restore it..." << std::endl;
	ext.restore_nraw();

	return 0;
}
