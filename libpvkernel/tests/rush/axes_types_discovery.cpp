#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVTypesDiscoveryOutput.h>
#include <pvkernel/rush/PVFileDescription.h>

#include "common.h"

#include <pvkernel/core/inendi_assert.h>

#include <cinttypes> // PRIxxx macro

#ifdef INSPECTOR_BENCH
constexpr static size_t nb_dup = 200;
#else
constexpr static size_t nb_dup = 1;
#endif

static constexpr const char* types_filename = TEST_FOLDER "/picviz/axes_types_discovery.csv";
static constexpr const char* types_format = TEST_FOLDER "/picviz/axes_types_discovery.csv.format";

static constexpr const char* time_filename = TEST_FOLDER "/picviz/time_types_discovery.csv";
static constexpr const char* time_format = TEST_FOLDER "/picviz/time_types_discovery.csv.format";

static const std::array<std::pair<std::string, std::vector<std::string>>, 8> AXES_TYPES{
    {{"number_int32", {""}},
     {"number_uint32", {"", "%#x", "%#o"}},
     {"number_float", {""}},
     {"number_double", {""}},
     {"ipv4", {""}},
     {"ipv6", {""}},
     {"mac_address", {""}},
     {"string", {""}}}};

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
	PVRush::PVTypesDiscoveryOutput types_discovery_output;
	double time =
	    discover_types(types_filename, PVRush::PVFormat("", types_format), types_discovery_output);

#ifndef INSPECTOR_BENCH
	PVCol col(0);
	for (const auto& formatter : AXES_TYPES) {
		const std::string formatter_name = formatter.first;
		for (const auto& formatter_params : formatter.second) {
			std::string type;
			std::string type_format;
			std::tie(type, type_format) = types_discovery_output.type_desc(col++);
			PV_VALID(type, formatter_name);
			PV_VALID(type_format, formatter_params);
		}
	}
#endif // INSPECTOR_BENCH

	PVRush::PVTypesDiscoveryOutput time_discovery_output;
	PVRush::PVFormat format("", time_format);
	time += discover_types(time_filename, format, time_discovery_output);

#ifndef INSPECTOR_BENCH
	for (PVCol col(0); col < (PVCol)format.get_storage_format().size(); col++) {
		PV_VALID(time_discovery_output.type_desc(col).first, std::string("time"));
	}
#endif // INSPECTOR_BENCH

	std::cout << time;

	return 0;
}
