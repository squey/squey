#include <pvkernel/filter/PVFilterLibrary.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/filter/PVFieldSplitterChunkMatch.h>
#include "helpers.h"
#include "test-env.h"

#include <list>
#include <utility>
#include <iostream>
#include <cstdlib>
#include <algorithm>

#include <QCoreApplication>
#include <QString>

int main(int argc, char** argv)
{
	if (argc <= 1) {
		std::cerr << "Usage: " << argv[0] << " file" << std::endl;
		return 1;
	}

	init_env();
	QCoreApplication app(argc, argv);
	PVFilter::PVPluginsLoad::load_all_plugins();
	PVRush::PVPluginsLoad::load_all_plugins();

	// Input file
	QString path_file(argv[1]);
	PVCore::PVArgument file(path_file);

	// Get the source creator
	QString file_path(argv[1]);
	PVRush::PVSourceCreator_p sc_file;
	PVRush::PVFormat format;
	if (!PVRush::PVTests::get_file_sc(file, format, sc_file)) {
		return 1;
	}

	// Process that file with the found source creator thanks to the extractor
	PVRush::PVSourceCreator::source_p src = sc_file->create_source_from_input(file);
	if (!src) {
		std::cerr << "Unable to create PVRush source from file " << argv[1] << std::endl;
		return 1;
	}

	// Get a chunk
	PVCore::PVChunk* chunk = (*src)();

	// Guess the first splitter
	
	LIB_FILTER(PVFilter::PVFieldsFilter<PVFilter::one_to_many>)::list_filters const& lf = LIB_FILTER(PVFilter::PVFieldsFilter<PVFilter::one_to_many>)::get().get_list();
	LIB_FILTER(PVFilter::PVFieldsFilter<PVFilter::one_to_many>)::list_filters::const_iterator it;
	for (it = lf.begin(); it != lf.end(); it++) {
		PVFilter::PVFieldSplitterChunkMatch match(*it);
		match.push_chunk(chunk);

		PVCore::PVArgumentList args;
		size_t nfields;

		if (match.get_match(args, nfields)) {
			std::cout << "Filter " << qPrintable(it.key()) << " matches with " << nfields << " fields and arguments:" << std::endl;
			PVCore::dump_argument_list(args);
		}
	}

	return 0;
}
