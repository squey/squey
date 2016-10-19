/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <pvkernel/rush/PVTypesDiscoveryOutput.h>

#include <pvkernel/rush/PVNraw.h> // for PVNraw
#include <pvkernel/rush/PVFormat.h>

#include <pvkernel/core/PVChunk.h> // for PVChunk

#include <pvbase/types.h> // for PVRow

#include <pvcop/types/factory.h>
#include <boost/utility/string_ref.hpp>

static constexpr const char* USER_TIME_FORMATS_FILENAME = "user_time_formats.ini";

// clang-format off
static const PVRush::PVTypesDiscoveryOutput::autodet_type_t TYPES {{
//   {{ formatter, parameters}, { excluded_formatter_1, excluded_formatter_2, ...}}
     {{"number_uint32", ""},    { "number_float", "number_double", "time" }},
     {{"number_int32",  ""},    { "number_float", "number_double", "time" }},
     {{"number_uint32", "%#o"}, { "number_float", "number_double", "time" }},
     {{"number_uint32", "%#x"}, { "number_float", "number_double", "time" }},
     {{"number_float",  ""},    { "number_uint32", "number_int32", "time" }},
     {{"number_double", ""},    { "number_uint32", "number_int32", "time" }},

     {{"ipv4",          ""},    {}},
     {{"ipv6",          ""},    {}},
     {{"mac_address",   ""},    {}},
}};

static const std::vector<std::string> SUPPLIED_TIMES_FORMATS {{
	"yyyy.MMMM.dd H:mm",
	"yyyy-M-d h:mm",
	"yyyy-M-d H:m:s",
	"yyyy-M-d H:m:ss.S",
	"yyyy/MM/dd HH:mm:ss Z",
	"yyyy/M/d",
	"MMM d H:m:s",
	"MMM d yyyy H:m:s",
	"eee MMM d H:m:ss yyyy",
	"dMMMyyyy H:m:s",
	"dMMMyyyy",
	"d/M/yy H:m:s",
	"H:m:s",
	"H%m%s",
	"d/M/yyyy H:m:s.S",
	"yy H:m:s.S",
	"yy H%m%s.S",
	"yy-M-d H:mm:ss.SSS",
	"yy-M-d H:mm:ss.SSS V",
}};
// clang-format on

static std::string get_user_time_formats_path()
{
	return std::string(getenv("HOME")) + "/" + INENDI_INSPECTOR_CONFDIR + "/" +
	       USER_TIME_FORMATS_FILENAME;
}

static PVRush::PVTypesDiscoveryOutput::autodet_type_t supported_types()
{
	PVRush::PVTypesDiscoveryOutput::autodet_type_t types = TYPES;

	// supplied time formats
	for (const std::string& suplied_time_format : SUPPLIED_TIMES_FORMATS) {
		types.push_back({{"time", suplied_time_format}, {}});
	}

	// user time formats (update file if needed to remove potential duplications)
	std::string updated_user_time_formats;
	const std::unordered_set<std::string> supplied_times_formats(SUPPLIED_TIMES_FORMATS.begin(),
	                                                             SUPPLIED_TIMES_FORMATS.end());
	std::ifstream in_f(get_user_time_formats_path());
	bool user_time_formats_file_needs_update = false;
	for (std::string user_time_format; getline(in_f, user_time_format);) {
		auto it = supplied_times_formats.find(user_time_format);
		bool duplicated_time_format = it != supplied_times_formats.end();
		user_time_formats_file_needs_update |= duplicated_time_format;
		if (not duplicated_time_format) {
			types.push_back({{"time", user_time_format}, {}});
			updated_user_time_formats += user_time_format + "\n";
		}
	}
	if (user_time_formats_file_needs_update) {
		in_f.close();
		std::ofstream out_f(get_user_time_formats_path());
		out_f << updated_user_time_formats;
	}

	return types;
}

static pvcop::formatter_desc get_formatter_desc(const std::string& type,
                                                const std::string& type_format)
{
	if (type == "time") {
		return PVRush::PVFormat::get_datetime_formatter_desc(type_format);
	}

	return {type, type_format};
}

void PVRush::PVTypesDiscoveryOutput::append_time_formats(
    const std::unordered_set<std::string>& time_formats)
{
	std::ofstream f;
	f.open(get_user_time_formats_path(), std::ios_base::app | std::ios_base::out);
	for (const std::string& time_format : time_formats) {
		f << time_format << std::endl;
	}
}

std::unordered_set<std::string> PVRush::PVTypesDiscoveryOutput::supported_time_formats()
{
	std::unordered_set<std::string> supported_time_formats;

	// supplied time formats
	for (const std::string& suplied_time_format : SUPPLIED_TIMES_FORMATS) {
		supported_time_formats.emplace(suplied_time_format);
	}

	// user time formats
	std::string time_format;
	std::ifstream f(get_user_time_formats_path());
	for (std::string supported_time_format; getline(f, supported_time_format);) {
		supported_time_formats.emplace(supported_time_format);
	}
	f.close();

	return supported_time_formats;
}

void PVRush::PVTypesDiscoveryOutput::prepare_load(const PVRush::PVFormat& format)
{
	_types = supported_types();
	_column_count = format.get_storage_format().size();

	for (const auto& type : _types) {
		pvcop::formatter_desc fd = get_formatter_desc(type.first.first, type.first.second);
		_formatters.emplace_back(pvcop::types::factory::create(fd.name(), fd.parameters()));
	}

	_matching_formatters =
	    matching_formatters_t(_column_count, std::vector<bool>(_formatters.size(), true));
}

void PVRush::PVTypesDiscoveryOutput::operator()(PVCore::PVChunk* chunk)
{
	assert(_matching_formatters.size() == _column_count);

	matching_formatters_t matching_formatters(_column_count,
	                                          std::vector<bool>(_formatters.size(), true));

	PVCore::list_elts const& elts = chunk->c_elements();

	PVRow local_row = 0;
	for (PVCore::PVElement* elt : elts) {

		PVCore::PVElement& e = *elt;
		assert(not e.filtered() and "We can't have filtered value in the Nraw");
		if (!e.valid()) {
			continue;
		}

		PVCore::list_fields const& fields = e.c_fields();
		PVCol col = 0;
		for (PVCore::PVField const& field : fields) {
			for (size_t idx = 0; idx < _formatters.size(); idx++) {
				if (matching_formatters[col][idx]) {
					pvcop::db::uint128_t t;
					std::string f(field.begin(), field.end());
					bool res = f.empty() or _formatters[idx]->from_string(f.c_str(), &t, 0);
					matching_formatters[col][idx] = matching_formatters[col][idx] & res;

					/**
					 * Disable mutually exclusive formatters to speed-up autodetection
					 */
					if (local_row == 0 && matching_formatters[col][idx]) {
						for (size_t i = 0; i < _formatters.size(); i++) {
							for (const std::string& excl_fmt : _types[idx].second) {
								if (strcmp(_formatters[i]->name(), excl_fmt.c_str()) == 0) {
									matching_formatters[col][i] = false;
								}
							}
						}
					}
				}
			}
			col++;
		}
		local_row++;
	}

// merge structs
#pragma omp critical
	{
		for (size_t col = 0; col < _column_count; col++) {
			for (size_t idx = 0; idx < _formatters.size(); idx++) {
				_matching_formatters[col][idx] =
				    _matching_formatters[col][idx] & matching_formatters[col][idx];
			}
		}
		_row_count += elts.size();
		_out_size += chunk->get_init_size();
	}

	chunk->free();
}

void PVRush::PVTypesDiscoveryOutput::job_has_finished(const PVControllerJob::invalid_elements_t&)
{
	assert(_matching_formatters.size() == _column_count);

	for (size_t col = 0; col < _column_count; col++) {
		bool formatter_found = false;
		for (size_t idx = 0; idx < _formatters.size(); idx++) {
			if (_matching_formatters[col][idx]) {
				_types_desc.emplace_back(_types[idx].first.first, _types[idx].first.second);
				formatter_found = true;
				break;
			}
		}
		if (not formatter_found) {
			_types_desc.emplace_back("string", ""); // default to string type
		}
	}
}

PVRush::PVTypesDiscoveryOutput::type_desc_t
PVRush::PVTypesDiscoveryOutput::type_desc(PVCol col) const
{
	assert(col < (PVCol)_types_desc.size());

	return _types_desc[col];
}
