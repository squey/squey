/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/rush/PVNraw.h>
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
	PVRush::PVInputType_p file_type =
	    LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("file");

	// Auto-discovery on that input
	std::multimap<float, PVRush::pair_format_creator> discovery =
	    PVRush::PVSourceCreatorFactory::discover_input(file_type, file);
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

	PVLOG_INFO("Source created.\n");

	PVRush::PVNraw nraw;
	PVRush::PVExtractor ext(used_format, nraw, sc_file, {file});
	PVRush::PVControllerJob_p job = ext.process_from_agg_nlines(0);
	job->wait_end();

	return 0;
}
