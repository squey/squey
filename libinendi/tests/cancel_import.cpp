/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVNrawOutput.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVAggregator.h>
#include <inendi/PVSource.h>

#include "common.h"

#include <pvkernel/core/inendi_assert.h>

#include <boost/thread.hpp>
#include <time.h>

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " file format" << std::endl;
		return 1;
	}

	static const char* input_file = argv[1];
	static const char* format_file = argv[2];

	init_env();

	PVRush::PVNraw nraw;
	PVRush::PVNrawOutput nraw_output(nraw);
	QList<std::shared_ptr<PVRush::PVInputDescription>> inputs;
	inputs << PVRush::PVInputDescription_p(
	    new PVRush::PVFileDescription(QString::fromStdString(input_file)));
	PVRush::PVSourceCreator_p sc_file;
	PVRush::PVFormat format("", format_file);
	PVRush::PVTests::get_file_sc(inputs.front(), format, sc_file);
	PVRush::PVSourceDescription src_desc(inputs, sc_file, format);

	Inendi::PVRoot root;
	Inendi::PVScene& scene = root.emplace_add_child("scene");
	Inendi::PVSource& source = scene.emplace_add_child(src_desc);

	PVRush::PVControllerJob_p job_import = source.extract(0);
	boost::this_thread::sleep(boost::posix_time::milliseconds(100));

	PV_VALID(job_import->running(), true);

	job_import->cancel();

	PV_VALID(job_import->running(), false);

	source.wait_extract_end(job_import);

	return 0;
}
