#define SIMULATE_PIPELINE
#include <pvfilter/PVChunkFilter.h>
#include <pvfilter/PVChunkFilterByElt.h>
#include <pvfilter/PVElementFilterByFields.h>
#include <pvfilter/PVPluginsLoad.h>
#include <pvrush/PVPluginsLoad.h>
#include <pvrush/PVInputFile.h>
#include <pvrush/PVUnicodeSource.h>
#include <pvrush/PVSourceCreator.h>
#include <pvrush/PVSourceCreatorFactory.h>
#include <pvrush/PVExtractor.h>
#include <cstdlib>
#include <iostream>
#include "helpers.h"
#include <QCoreApplication>
#include "test-env.h"

using std::cout;
using std::cerr;
using std::endl;

using namespace PVRush;
using namespace PVCore;

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
	PVFilter::PVArgument file = PVFilter::PVArgument(QString(argv[1]));

	// Load the given format file
	PVFormat format("format", argv[2]);
	if (!format.populate()) {
		std::cerr << "Can't read format file " << argv[2] << std::endl;
		return 1;
	}

	// Load source plugins that take a file as input
	PVRush::PVInputType_p in_t = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("file");
	if (!in_t) {
		std::cerr << "Unable to load the file input type plugin !" << std::endl;
		return 1;
	}
	PVRush::list_creators lcr = PVRush::PVSourceCreatorFactory::get_by_input_type(in_t);

	// Pre-discovery
	PVRush::list_creators::iterator itc;
	PVRush::list_creators pre_discovered_c;
	for (itc = lcr.begin(); itc != lcr.end(); itc++) {
		PVRush::PVSourceCreator_p sc = *itc;
		if (sc->pre_discovery(file)) {
			pre_discovered_c.push_back(sc);
		}
	}

	PVRush::PVSourceCreator_p sc_file;
	if (pre_discovered_c.size() == 0) {
		std::cerr << "No source plugins can open the file " << argv[1] << std::endl;
		return 1;
	}
	if (pre_discovered_c.size() == 1) {
		sc_file = *(pre_discovered_c.begin());
	}
	else {
		// Take the source creator that have the highest success rate with the given format
		float success_rate = -1;
		for (itc = pre_discovered_c.begin(); itc != pre_discovered_c.end(); itc++) {
			PVRush::PVSourceCreator_p sc = *itc;
			PVRush::pair_format_creator fcr(format, sc);
			float sr_tmp = PVRush::PVSourceCreatorFactory::discover_input(fcr, file);
			if (sr_tmp > success_rate) {
				success_rate = sr_tmp;
				sc_file = sc;
			}
		}
		std::cerr << "Chose source creator '" << qPrintable(sc_file->name()) << "' with success rate of " << success_rate << std::endl;
	}

	
	// Process that file with the found source creator thanks to the extractor
	PVRush::PVSourceCreator::source_p src = sc_file->create_source_from_input(file);
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
