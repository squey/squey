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
#include <algorithm>
#include <array>
#include <cctype>
#include <functional>
#include <optional>
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

static const std::pair<std::string, std::string> DECIMAL_COMMA{"number_double", ","};

/**
 * Whether @p value writes a decimal with a comma, rather than thousands, as "1,234" may
 */
static bool decimal_comma(std::string_view value)
{
	const size_t comma = value.find(',');
	if (comma == std::string_view::npos) {
		return false;
	}
	const size_t end = value.find_first_not_of("0123456789", comma + 1);
	return (end == std::string_view::npos ? value.size() : end) - comma - 1 != 3;
}

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
     // display 25.0100002. Its default notation writes the fewest digits that read back as
     // the value, which are those of the file.
     {"number_double", ""},
     // the decimal comma of the files whose separator is not one: see prepare_load()
     {"number_double", ","},

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
	"yyyy-M-d'T'H:m:s'Z'",
	"yyyy/M/d H:m:s",
	// ISO 8601, and the timestamps with a zone of PostgreSQL. The formats with a fraction
	// of a second read a "T" as well as a space, as "yyyy-M-d H:m:ss.S" does.
	"yyyy-M-d",
	"yyyy-M-d'T'H:m:s",
	"yyyy-M-d'T'H:m:sX",
	"yyyy-M-d H:m:sX",
	"yyyy-M-d H:m:s.SX",
	// the month first, which a date only reads as when it does not read day first
	"M/d/yyyy",
	"M/d/yyyy H:m:s",
	"M/d/yy",
	"M/d/yy H:m:s",
	// the logs of web servers, and the dates of mails and HTTP
	"d/MMM/yyyy:H:m:s Z",
	"eee, d MMM yyyy H:m:s Z"
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
	 *
	 * The formats otherwise keep the order of the list, whatever the standard library: a
	 * date such as "03/04/2024" reads day first as well as month first, and is read the
	 * way the list gives first.
	 */
	std::stable_sort(time_formats.begin(), time_formats.end(), [](const auto& p1, const auto& p2) {
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

/**
 * Whether @p value is one of the ways files write that a field has no value
 */
static bool says_missing(std::string_view value)
{
	static constexpr std::array<std::string_view, 9> missing{"-",   "null", "none", "nil",   "na",
	                                                         "n/a", "nan",  "#n/a", "(null)"};

	const size_t first = value.find_first_not_of(" \t");
	if (first == std::string_view::npos) {
		return false;
	}
	value = value.substr(first, value.find_last_not_of(" \t") - first + 1);

	return std::ranges::any_of(missing, [value](std::string_view word) {
		return std::ranges::equal(value, word, [](char a, char b) {
			return std::tolower(static_cast<unsigned char>(a)) == b;
		});
	});
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
	// a comma in a value of a file it separates the values of is no decimal one
	const std::optional<PVRush::PVCSVQuoting> quoting = format.csv_quoting();
	if (quoting and quoting->separator == ',') {
		std::erase(_types, DECIMAL_COMMA);
	}
	_column_count = format.get_storage_format().size();
	_names.resize(_column_count);

	for (const auto& type : _types) {
		pvcop::formatter_desc fd = get_formatter_desc(type.first, type.second);
		_formatters.emplace_back(pvcop::types::factory::create(fd.name(), fd.parameters()));
		_notations.push_back(notation_of(type));
	}
	const auto decimal_comma_type = std::ranges::find(_types, DECIMAL_COMMA);
	_decimal_comma_index = decimal_comma_type == _types.end()
	                           ? std::numeric_limits<size_t>::max()
	                           : size_t(decimal_comma_type - _types.begin());

	_matching_formatters =
	    matching_formatters_t(_column_count, std::vector<bool>(_formatters.size(), true));
	_valued = std::vector<bool>(_column_count, false);
	_decimal_commas = _valued;
	_first_row.clear();
	_first_row_matching = _matching_formatters;
	_first_row_valued = _valued;
}

void PVRush::PVTypesDiscoveryOutput::operator()(PVCore::PVChunk* c)
{
	auto* chunk = dynamic_cast<PVCore::PVTextChunk*>(c);
	assert(chunk);
	assert(_matching_formatters.size() == _column_count);

	bool first_chunk = chunk->index() == 0;

	matching_formatters_t matching_formatters(_column_count,
	                                          std::vector<bool>(_formatters.size(), true));
	std::vector<bool> valued(_column_count, false);
	std::vector<bool> decimal_commas(_column_count, false);
	matching_formatters_t first_row_matching;
	std::vector<bool> first_row_valued;
	std::vector<std::string> first_row;

	PVCore::list_elts const& elts = chunk->c_elements();

	PVRow local_row = 0;
	std::string value;
	for (PVCore::PVElement* elt : elts) {

		PVCore::PVElement& e = *elt;
		assert(not e.filtered() and "We can't have filtered value in the Nraw");
		if (!e.valid()) {
			continue;
		}

		const bool is_first_row = first_chunk and local_row == 0;
		if (is_first_row) {
			first_row_matching = matching_formatters;
			first_row_valued = valued;
			first_row.resize(_column_count);
		}
		matching_formatters_t& matching = is_first_row ? first_row_matching : matching_formatters;
		std::vector<bool>& row_valued = is_first_row ? first_row_valued : valued;

		PVCore::list_fields const& fields = e.c_fields();
		PVCol col(0);
		for (PVCore::PVField const& field : fields) {
			value.assign(field.begin(), field.end());
			if (is_first_row) {
				first_row[col] = value;
			}
			// A field empty, or saying it has no value, rules no type out: "-" in a column of
			// sizes, "NULL" in a column of dates.
			if (not value.empty() and not says_missing(value)) {
				row_valued[col] = true;
				for (size_t idx = 0; idx < _formatters.size(); idx++) {
					if (not matching[col][idx]) {
						continue;
					}
					if (not written_as(_notations[idx], value)) {
						matching[col][idx] = false;
						continue;
					}
					pvcop::db::uint128_t t;
					bool pass_autodetect;
					_formatters[idx]->from_string(value.c_str(), &t, 0, &pass_autodetect);
					matching[col][idx] = pass_autodetect;
				}
				if (_decimal_comma_index < _formatters.size() and
				    matching[col][_decimal_comma_index] and decimal_comma(value)) {
					decimal_commas[col] = true;
				}
			}
			col++;
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
			_valued[col] = _valued[col] or valued[col];
			_decimal_commas[col] = _decimal_commas[col] or decimal_commas[col];
		}
		if (not first_row.empty()) {
			_first_row = std::move(first_row);
			_first_row_matching = std::move(first_row_matching);
			_first_row_valued = std::move(first_row_valued);
		}
		_row_count += elts.size();
		_out_size += chunk->get_init_size();
	}

	chunk->free();
}

void PVRush::PVTypesDiscoveryOutput::job_has_finished(const PVControllerJob::invalid_elements_t&)
{
	assert(_matching_formatters.size() == _column_count);

	// the type of a column, of those after the first row: none for a column with no value
	// in the sample, which has no type to read it with but text
	const auto type_of = [this](size_t col) -> std::optional<size_t> {
		if (not _valued[col]) {
			return {};
		}
		for (size_t idx = 0; idx < _formatters.size(); idx++) {
			if (_matching_formatters[col][idx] and
			    (idx != _decimal_comma_index or _decimal_commas[col])) {
				return idx;
			}
		}
		return {};
	};
	bool not_all_strings = false;
	for (size_t col = 0; col < _column_count; col++) {
		not_all_strings = not_all_strings or type_of(col);
	}

	// The first row names the columns when it holds no value a type reads, as names
	// rarely are numbers or dates, or when a column of numbers or dates does not read the
	// value it holds there: "id" above integers, or "2024" above the decimals of a year.
	// The values it holds nothing in, or the "-" or "NULL" of a missing value, tell nothing.
	bool names_only = true;
	bool named = false;
	bool inconsistent = false;
	for (size_t col = 0; col < _first_row.size(); col++) {
		if (not _first_row_valued[col]) {
			continue;
		}
		named = true;
		names_only = names_only and std::ranges::none_of(_first_row_matching[col], std::identity());
		const std::optional<size_t> type = type_of(col);
		inconsistent = inconsistent or (type and not _first_row_matching[col][*type]);
	}
	const bool has_header = named and not_all_strings and (names_only or inconsistent);

	if (has_header) {
		_names = _first_row;
	} else {
		for (size_t col = 0; col < _first_row.size(); col++) {
			if (_first_row_valued[col]) {
				_valued[col] = true;
				for (size_t idx = 0; idx < _formatters.size(); idx++) {
					_matching_formatters[col][idx] =
					    _matching_formatters[col][idx] and _first_row_matching[col][idx];
				}
			}
		}
	}
	for (size_t col = 0; col < _column_count; col++) {
		std::string col_name =
		    has_header ? ((col == 0 and not _names[col].empty() and _names[col][0] == '#')
		                      ? _names[col].substr(1, _names[col].size() - 1)
		                      : _names[col])
		               : "";

		if (const std::optional<size_t> type = type_of(col)) {
			_types_desc.emplace_back(_types[*type].first, _types[*type].second, col_name);
		} else {
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
