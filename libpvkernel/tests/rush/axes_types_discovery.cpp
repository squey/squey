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
#include <pvkernel/rush/PVTypesDiscoveryOutput.h>
#include <pvkernel/rush/PVFileDescription.h>

#include "common.h"

#include <pvkernel/core/squey_assert.h>

#include <cinttypes> // PRIxxx macro

#ifdef SQUEY_BENCH
constexpr static size_t nb_dup = 200;
#else
constexpr static size_t nb_dup = 1;
#endif

static constexpr const char* types_filename = TEST_FOLDER "/picviz/axes_types_discovery.csv";
static constexpr const char* types_format = TEST_FOLDER "/picviz/axes_types_discovery.csv.format";

static constexpr const char* time_filename = TEST_FOLDER "/picviz/time_types_discovery.csv";
static constexpr const char* time_format = TEST_FOLDER "/picviz/time_types_discovery.csv.format";

static double discover_types(const std::string& filename,
                             const PVRush::PVFormat& format,
                             PVRush::PVTypesDiscoveryOutput& type_discovery_output)
{
	QList<std::shared_ptr<PVRush::PVInputDescription>> list_inputs;
	for (size_t dup = 0; dup < nb_dup; dup++) {
		list_inputs << PVRush::PVInputDescription_p(
		    new PVRush::PVFileDescription(QString::fromStdString(filename)));
	}
	PVRush::PVSourceCreator_p sc_file;
	PVRush::PVTests::get_file_sc(list_inputs.front(), format, sc_file);
	PVRush::PVExtractor extractor(format, type_discovery_output, sc_file, list_inputs);

	auto start = std::chrono::system_clock::now();

	PVRush::PVControllerJob_p job =
	    extractor.process_from_agg_idxes(0, IMPORT_PIPELINE_ROW_COUNT_LIMIT);
	job->wait_end();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	return diff.count();
}

int main()
{
	pvtest::init_ctxt();

	PVRush::PVFormat types_fmt("", types_format);

	PVRush::PVTypesDiscoveryOutput types_discovery_output;
	double time = discover_types(types_filename, types_fmt, types_discovery_output);

#ifndef SQUEY_BENCH
	PVCol col(0);
	(void) col;
	for (PVCol col(0); col < (PVCol)types_fmt.get_storage_format().size(); col++) {
		std::string type;
		std::string type_format;
		std::string axe_name;
		std::tie(type, type_format, axe_name) = types_discovery_output.type_desc(col);
		PV_VALID(type + type_format, axe_name);
	}
#endif // SQUEY_BENCH

	PVRush::PVTypesDiscoveryOutput time_discovery_output;
	PVRush::PVFormat time_fmt("", time_format);
	time += discover_types(time_filename, time_fmt, time_discovery_output);

#ifndef SQUEY_BENCH
	std::vector<std::string> axes_types(time_fmt.get_axes().size());
	std::transform(time_fmt.get_axes().begin(), time_fmt.get_axes().end(), axes_types.begin(),
	               [](const PVRush::PVAxisFormat& axis) {
		               return axis.get_name().split("_")[0].toStdString();
		           });
	for (PVCol col(0); col < (PVCol)time_fmt.get_storage_format().size(); col++) {
		std::string type;
		std::tie(type, std::ignore, std::ignore) = time_discovery_output.type_desc(col);
		PV_VALID(type, axes_types[col]);
	}
#endif // SQUEY_BENCH

	std::cout << time;

	return 0;
}
