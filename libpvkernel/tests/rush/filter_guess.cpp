/**
 * \file filter_guess.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVFileDescription.h>
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

#include <QString>

int main(int argc, char** argv)
{
	if (argc <= 1) {
		std::cerr << "Usage: " << argv[0] << " file" << std::endl;
		return 1;
	}

	init_env();
	PVFilter::PVPluginsLoad::load_all_plugins();
	PVRush::PVPluginsLoad::load_all_plugins();

	// Input file
	QString path_file(argv[1]);
	PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(path_file));

	// Get the source creator
	QString file_path(argv[1]);
	PVRush::PVSourceCreator_p sc_file = LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("text_file");
	if (!sc_file) {
		std:: cerr << "text_file source creator plugin isn't available!" << std::endl;
		return 1;
	}

	PVRush::PVFormat format;
	// Process that file with the found source creator thanks to the extractor
	PVRush::PVSourceCreator::source_p src = sc_file->create_discovery_source_from_input(file, format);
	if (!src) {
		std::cerr << "Unable to create PVRush source from file " << argv[1] << std::endl;
		return 1;
	}

	// Get a chunk
	PVCore::PVChunk* chunk = (*src)();

	// Guess the first splitter
	
	LIB_CLASS(PVFilter::PVFieldsFilter<PVFilter::one_to_many>)::list_classes const& lf = LIB_CLASS(PVFilter::PVFieldsFilter<PVFilter::one_to_many>)::get().get_list();
	LIB_CLASS(PVFilter::PVFieldsFilter<PVFilter::one_to_many>)::list_classes::const_iterator it;
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
