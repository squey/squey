
#include "common_guess.h"

#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVFileDescription.h>

#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/filter/PVFieldSplitterChunkMatch.h>

#include <iostream>

PVFilter::PVFieldsSplitter_p pvtest::guess_filter(const char* filename, PVCol& axes_count)
{
	PVFilter::PVPluginsLoad::load_all_plugins();
	PVRush::PVPluginsLoad::load_all_plugins();

	// Input file
	QString path_file(filename);
	PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(path_file));

	// Get the source creator
	PVRush::PVSourceCreator_p sc_file =
	    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("text_file");

	PVRush::PVFormat format;
	// Process that file with the found source creator thanks to the extractor
	PVRush::PVSourceCreator::source_p src = sc_file->create_source_from_input(file, format);
	if (!src) {
		return PVFilter::PVFieldsSplitter_p();
	}

	return PVFilter::PVFieldSplitterChunkMatch::get_match_on_input(src, axes_count);
}
