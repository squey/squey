//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVNrawOutput.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVAggregator.h>
#include <squey/PVSource.h>

#include "common.h"

#include <pvkernel/core/squey_assert.h>

#include <boost/thread.hpp>
#include <ctime>

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

	Squey::PVRoot root;
	Squey::PVScene& scene = root.emplace_add_child("scene");
	Squey::PVSource& source = scene.emplace_add_child(src_desc);

	PVRush::PVControllerJob_p job_import = source.extract(0);
	boost::this_thread::sleep(boost::posix_time::milliseconds(100));

	PV_VALID(job_import->running(), true);

	job_import->cancel();

	PV_VALID(job_import->running(), false);

	source.wait_extract_end(job_import);

	return 0;
}
