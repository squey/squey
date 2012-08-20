#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/rush/PVFileDescription.h>

#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>
#include <picviz/PVSource.h>
#include <picviz/PVView.h>

#include "common.h"

Picviz::PVSource_sp get_src_from_file(Picviz::PVScene_sp scene, QString const& path_file, QString const& path_format)
{
	// Input file
	PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(path_file));

	// Load the given format file
	PVRush::PVFormat format("format", path_format);
	if (!format.populate()) {
		std::cerr << "Can't read format file " << qPrintable(path_format) << std::endl;
		return Picviz::PVSource_p::invalid();
	}   

	PVRush::PVSourceCreator_p sc_file;
	if (!PVRush::PVTests::get_file_sc(file, format, sc_file)) {
		return Picviz::PVSource_p::invalid();
	}   

	Picviz::PVSource_p src(scene, PVRush::PVInputType::list_inputs() << file, sc_file, format);
	src->get_extractor().get_agg().set_strict_mode(true);
	PVRush::PVControllerJob_p job = src->extract_from_agg_nlines(0, 200000);
	job->wait_end();

	return src;
}

Picviz::PVSource_sp get_src_from_file(Picviz::PVRoot_sp root, QString const& file, QString const& format)
{
	Picviz::PVScene_p scene(root);
	return get_src_from_file(Picviz::PVScene_sp(scene), file, format);
}
