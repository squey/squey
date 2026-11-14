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

#include <pvkernel/rush/PVTypesDiscoveryOutput.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/core/PVTextChunk.h> // for PVChunk
#include <pvkernel/core/PVConfig.h>
#include <pvbase/types.h> // for PVRow
#include <pvcop/types/factory.h>
#include <cassert> // for assert
#include <atomic>
#include <fstream>
#include <list>
#include <mutex>
#include <cinttypes> // PRIxxx macro
#include <cctype>
#include <string_view>

#include "pvcop/db/128bits_support.h"
#include "pvcop/formatter_desc.h"
#include "pvcop/formatter_desc_list.h"
#include "pvkernel/core/PVChunk.h"
#include "pvkernel/core/PVElement.h"
#include "pvkernel/core/PVField.h"
#include "pvkernel/rush/PVControllerJob.h"
#include "type_safe/strong_typedef.hpp"

static constexpr const char* USER_TIME_FORMATS_FILENAME = "user_time_formats.ini";

// clang-format off
static const PVRush::PVTypesDiscoveryOutput::autodet_type_t TYPES {{
//    { formatter, parameters }
     {"number_uint32", ""},
     {"number_int32",  ""},
     {"number_uint32", "%#" PRIo32},
     {"number_uint32", "0x%" PRIx32},
     {"number_uint64", ""},
     {"number_int64",  ""},
     {"number_uint64", "%#" PRIo64},
     {"number_uint64", "0x%" PRIx64},
     // A double, as a float keeps 24 bits: "4326096.668" would read 4326096.5, and "25.01"
     // display 25.0100002. Fifteen digits are what a double writes back as written: its
     // default seventeen display "25.01" as 25.010000000000002.
     {"number_double", "%.15g"},

     {"ipv4",          ""},
     {"ipv6",          ""},
     {"mac_address",   ""},
     {"duration",      ""},
}};

static const std::vector<std::string> SUPPLIED_TIMES_FORMATS {{
	"yyyy.MMMM.dd H:mm",
	"yyyy.MMM.d H:mm.ss",
	"yyyy-M-d h:mm",
	"yyyy-M-d H:m:s",
	"yyyy/MM/dd HH:mm:ss Z",
	"MMM d H:m:s",
	"MMM d yyyy H:m:s",
	"eee MMM d H:m:ss yyyy",
	"dMMMyyyy H:m:s",
	"dMMMyyyy",
	"d-M-yyyy",
	"d-M-yy",
	"d-M-yy H:m:s",
	"d-M-yyyy H:m:s",
	"d/M/yyyy",
	"d/M/yy",
	"d/M/yy H:m:s",
	"d/M/yyyy H:m:s",
	"dd.MM.yyyy",
	"dd.MM.yyyy H:m:s",
	"dd.MM.yy",
	"dd.MM.yy H:m:s",
	"yyyy/M/d",
	"d/M/yy H:m:s.S",
	"d/M/yyyy H:m:s.S",
	"d-M-yy H:m:s.S",
	"d-M-yyyy H:m:s.S",
	"yy H:m:s.S",
	"yy H%m%s.S",
	"yyyy-M-d H:m:ss.S",
	"yy-M-d H:mm:ss.SSS",
	"yy-M-d H:mm:ss.SSS V",
	"d-M-yy H:m:s:S",
	"dd.MM.yyyy H:m:s.S",
	"dd.MM.yy H:m:s.S",
	"yyyy-M-d'T'H:m:s'Z'"
}};
// clang-format on

std::mutex g_mutex;

static QString get_user_time_formats_path()
{
	return PVCore::PVConfig::user_dir() + USER_TIME_FORMATS_FILENAME;
}

static std::vector<std::string> sort_time_formats_to_reduce_false_detection_rate(
    const std::vector<std::string>& supplied_time_formats)
{
	std::vector<std::string> time_formats = supplied_time_formats;

	/* Put times formats containing 4 year digits ('yyyy') at the top of the list
	 *
	 * Indeed, when a 2 digit year is specified in the time format but 4
	 * digits are actually provided in the time string, libc and boost are
	 * silently interpreting the first 2 digits as the year.
	 *
	 * eg : "d/M/yy" "04/11/2016", year is interpreted as "20" meaning 1900 + 20 = 1920.
	 *
	 * As this problem doesn't occur the other way around (specifing 4 year digits in the
	 * format but only providing 2), according a greater priority to the 4 year digits solves
	 * this problem for the types autodetection.
	 */
	std::sort(time_formats.begin(), time_formats.end(), [](const auto& p1, const auto& p2) {
		bool p1_4_years_digit = p1.find("yyyy") != std::string::npos;
		bool p2_4_years_digit = p2.find("yyyy") != std::string::npos;

		return p1_4_years_digit and not p2_4_years_digit;
	});

	return time_formats;
}

static PVRush::PVTypesDiscoveryOutput::autodet_type_t supported_types()
{
	PVRush::PVTypesDiscoveryOutput::autodet_type_t types = TYPES;
	std::vector<std::string> time_formats = SUPPLIED_TIMES_FORMATS;

	// user time formats (update file if needed to remove potential duplications)
	QString updated_user_time_formats;
	const std::unordered_set<std::string> supplied_times_formats(SUPPLIED_TIMES_FORMATS.begin(),
	                                                             SUPPLIED_TIMES_FORMATS.end());
    QFile file(get_user_time_formats_path());
	bool user_time_formats_file_needs_update = false;
	if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream in(&file);
        while (!in.atEnd()) {
            QString user_time_format = in.readLine();
            bool duplicated = supplied_times_formats.contains(user_time_format.toStdString());
            user_time_formats_file_needs_update |= duplicated;
            if (!duplicated) {
                time_formats.push_back(user_time_format.toStdString());
                updated_user_time_formats += user_time_format + '\n';
            }
        }
        file.close();
    }

    if (user_time_formats_file_needs_update) {
        if (file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) {
            QTextStream out(&file);
            out << updated_user_time_formats;
            file.close();
        }
    }

	time_formats = sort_time_formats_to_reduce_false_detection_rate(time_formats);
	for (const std::string& time_format : time_formats) {
		types.push_back({"time", time_format});
	}

	return types;
}

/**
 * Whether @p value is written as numbers of the notation @p n are written back.
 *
 * The number formatters read more than they write: an integer type reads the zip code
 * "01234" and writes it back as 1234, the octal one reads "75001", which it writes back
 * as 075001, and the hexadecimal one reads words such as "cafe" or "d", which it writes
 * back with a "0x" they never had. They skip leading blanks and a sign, and so does this.
 */
static bool written_as(PVRush::PVTypesDiscoveryOutput::notation n, std::string_view value)
{
	using notation = PVRush::PVTypesDiscoveryOutput::notation;

	const size_t start = value.find_first_not_of(" \t");
	if (start == std::string_view::npos) {
		return true;
	}
	value.remove_prefix(start);
	if (value.starts_with('+') or value.starts_with('-')) {
		value.remove_prefix(1);
	}

	switch (n) {
	case notation::decimal:
		return not(value.size() > 1 and value[0] == '0' and
		           std::isdigit(static_cast<unsigned char>(value[1])));
	case notation::octal:
		return value.starts_with('0');
	case notation::hexadecimal:
		return value.starts_with("0x") or value.starts_with("0X");
	case notation::none:
		break;
	}
	return true;
}

static PVRush::PVTypesDiscoveryOutput::notation
notation_of(const std::pair<std::string, std::string>& type)
{
	using notation = PVRush::PVTypesDiscoveryOutput::notation;

	if (not type.first.starts_with("number_")) {
		return notation::none;
	}
	// "0x%x", "0x%" PRIx64, "%#o", "%#" PRIo64
	if (type.second.find('x') != std::string::npos) {
		return notation::hexadecimal;
	}
	if (type.second.find('o') != std::string::npos) {
		return notation::octal;
	}
	return notation::decimal;
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
	QFile file(get_user_time_formats_path());
    if (file.open(QIODevice::Append | QIODevice::Text)) {
        QTextStream out(&file);
        for (const std::string& time_format : time_formats) {
            out << QString::fromStdString(time_format) << '\n';
        }
        file.close();
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
	QFile file(get_user_time_formats_path());
	if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
		QTextStream in(&file);
		while (!in.atEnd()) {
			QString supported_time_format = in.readLine();
			supported_time_formats.insert(supported_time_format.toStdString());
		}
		file.close();
	}

	return supported_time_formats;
}

void PVRush::PVTypesDiscoveryOutput::prepare_load(const PVRush::PVFormat& format)
{
	_types = supported_types();
	_column_count = format.get_storage_format().size();
	_names.resize(_column_count);

	for (const auto& type : _types) {
		pvcop::formatter_desc fd = get_formatter_desc(type.first, type.second);
		_formatters.emplace_back(pvcop::types::factory::create(fd.name(), fd.parameters()));
		_notations.push_back(notation_of(type));
	}

	_matching_formatters =
	    matching_formatters_t(_column_count, std::vector<bool>(_formatters.size(), true));
}

void PVRush::PVTypesDiscoveryOutput::operator()(PVCore::PVChunk* c)
{
	auto* chunk = dynamic_cast<PVCore::PVTextChunk*>(c);
	assert(chunk);
	assert(_matching_formatters.size() == _column_count);

	bool first_chunk = chunk->index() == 0;

	matching_formatters_t matching_formatters(_column_count,
	                                          std::vector<bool>(_formatters.size(), true));

	PVCore::list_elts const& elts = chunk->c_elements();

	PVRow local_row = 0;
	std::string value;
	for (PVCore::PVElement* elt : elts) {

		PVCore::PVElement& e = *elt;
		assert(not e.filtered() and "We can't have filtered value in the Nraw");
		if (!e.valid()) {
			continue;
		}

		PVCore::list_fields const& fields = e.c_fields();
		PVCol col(0);
		for (PVCore::PVField const& field : fields) {
			value.assign(field.begin(), field.end());
			// an empty value rules no type out
			if (not value.empty()) {
				for (size_t idx = 0; idx < _formatters.size(); idx++) {
					if (not matching_formatters[col][idx]) {
						continue;
					}
					if (not written_as(_notations[idx], value)) {
						matching_formatters[col][idx] = false;
						continue;
					}
					pvcop::db::uint128_t t;
					bool pass_autodetect;
					_formatters[idx]->from_string(value.c_str(), &t, 0, &pass_autodetect);
					matching_formatters[col][idx] = pass_autodetect;
				}
			}
			col++;
		}

		// extract axes name for header
		if (first_chunk and local_row == 0) {
			bool is_header = std::all_of(
			    matching_formatters.begin(), matching_formatters.end(), [](const auto& f) {
				    return std::all_of(f.begin(), f.end(), [](bool v) { return not v; });
			    });
			if (is_header) {
				size_t col = 0;
				for (PVCore::PVField const& field : fields) {
					std::string f(field.begin(), field.end());
					_names[col] = f;
					std::fill(matching_formatters[col].begin(), matching_formatters[col].end(),
					          true);
					col++;
				}
			}
		}
		local_row++;
	}

	// merge structs
	std::lock_guard<std::mutex> guard(g_mutex);
	{
		for (size_t col = 0; col < _column_count; col++) {
			for (size_t idx = 0; idx < _formatters.size(); idx++) {
				_matching_formatters[col][idx] =
				    _matching_formatters[col][idx] && matching_formatters[col][idx];
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

	const auto& mf = _matching_formatters;
	bool not_all_strings = std::any_of(mf.begin(), mf.end(), [](const auto& f) {
		return std::any_of(f.begin(), f.end(), [](bool v) { return v; });
	});

	for (size_t col = 0; col < _column_count; col++) {

		bool formatter_found = false;
		std::string col_name =
		    not_all_strings ? ((col == 0 and not _names[col].empty() and _names[col][0] == '#')
		                           ? _names[col].substr(1, _names[col].size() - 1)
		                           : _names[col])
		                    : "";

		for (size_t idx = 0; idx < _formatters.size(); idx++) {
			if (_matching_formatters[col][idx]) {
				_types_desc.emplace_back(_types[idx].first, _types[idx].second, col_name);
				formatter_found = true;
				break;
			}
		}
		if (not formatter_found) {
			_types_desc.emplace_back("string", "", col_name); // default to string type
		}
	}
}

PVRush::PVTypesDiscoveryOutput::type_desc_t
PVRush::PVTypesDiscoveryOutput::type_desc(PVCol col) const
{
	assert(col < (PVCol)_types_desc.size());

	return _types_desc[col];
}
