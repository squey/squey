/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/rush/PVFileDescription.h>

#include <inendi/PVRoot.h>
#include <inendi/PVScene.h>
#include <inendi/PVSource.h>
#include <inendi/PVView.h>

#include "common.h"

Inendi::PVSource_sp
get_src_from_file(Inendi::PVScene_sp scene, QString const& path_file, QString const& path_format)
{
	// Input file
	PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(path_file));

	// Load the given format file
	PVRush::PVFormat format("format", path_format);
	if (!format.populate()) {
		throw std::runtime_error("Can't read format file " + path_format.toStdString());
	}

	PVRush::PVSourceCreator_p sc_file;
	if (!PVRush::PVTests::get_file_sc(file, format, sc_file)) {
		throw std::runtime_error("Can't read file source");
	}

	Inendi::PVSource_sp src =
	    scene->emplace_add_child(PVRush::PVInputType::list_inputs() << file, sc_file, format);
	PVRush::PVControllerJob_p job = src->extract(0, 200000);
	src->wait_extract_end(job);

	return src;
}

Inendi::PVSource_sp
get_src_from_file(Inendi::PVRoot_sp root, QString const& file, QString const& format)
{
	Inendi::PVScene_sp scene = root->emplace_add_child("scene");
	return get_src_from_file(scene, file, format);
}
