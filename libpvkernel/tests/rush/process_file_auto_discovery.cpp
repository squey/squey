#define SIMULATE_PIPELINE
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVSourceCreatorFactory.h>
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
	if (argc <= 1) {
		cerr << "Usage: " << argv[0] << " file" << endl;
		return 1;
	}

	init_env();
	QCoreApplication app(argc, argv);
	PVFilter::PVPluginsLoad::load_all_plugins();
	PVRush::PVPluginsLoad::load_all_plugins();

	// Input file
	QString path_file(argv[1]);
	PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(path_file));

	// Load source plugins that take a file as input
	PVRush::PVInputType_p file_type = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("file");
	if (!file_type) {
		std::cerr << "Unable to load the file input type plugin !" << std::endl;
		return false;
	}

	// Auto-discovery on that input
	std::multimap<float, PVRush::pair_format_creator> discovery = PVRush::PVSourceCreatorFactory::discover_input(file_type, file);
	std::multimap<float, PVRush::pair_format_creator>::const_iterator it;

	// Dump format statistics
	if (discovery.size() == 0) {
		PVLOG_WARN("No format have been discovered !\n");
		return 0;
	}
	for (it = discovery.begin(); it != discovery.end(); it++) {
		PVLOG_INFO("%s : %0.4f\n", qPrintable(it->second.first.get_format_name()), it->first);
	}

	// Process that file with the first format
	PVRush::pair_format_creator fc_first = discovery.rbegin()->second;
	PVRush::PVFormat used_format = fc_first.first;
	PVRush::PVSourceCreator_p sc_file = fc_first.second;
	used_format.populate();

	PVLOG_INFO("Creating source with format %s...\n", qPrintable(used_format.get_format_name()));
	PVRush::PVSourceCreator::source_p src = sc_file->create_source_from_input(file, used_format);
	if (!src) {
		std::cerr << "Unable to create PVRush source from file " << argv[1] << std::endl;
		return 1;
	}
	PVLOG_INFO("Source created.\n");

	PVRush::PVExtractor ext;
	ext.start_controller();
	ext.add_source(src);
	ext.set_format(used_format);
	ext.set_chunk_filter(used_format.create_tbb_filters());

	PVLOG_INFO("Asking 1 million lines...\n");
	// Ask for 1 million lines
	PVRush::PVControllerJob_p job = ext.process_from_agg_nlines(0, 1000000);
	job->wait_end();

	// Dump the NRAW to stdout
	dump_nraw_csv(ext.get_nraw());

	return 0;
}
