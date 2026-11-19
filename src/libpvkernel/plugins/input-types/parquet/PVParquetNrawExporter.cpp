/* * MIT License
 *
 * © Squey, 2026
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "PVParquetNrawExporter.h"

#include <pvcop/db/array.h>
#include <pvcop/db/read_dict.h>
#include <pvcop/db/string_index_types.h>

#include <arrow/api.h>
#include <parquet/arrow/writer.h>

#include <boost/date_time/gregorian/gregorian_types.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

#include <tbb/parallel_for.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace
{

/** The least count of rows written at once, but for the last ones. */
constexpr const size_t BATCH_ROW_COUNT = 64 * 1024;

using rows_t = std::vector<PVRow>;
using array_of_t =
    std::function<std::shared_ptr<arrow::Array>(const pvcop::db::array&, const rows_t&)>;

/** How the values of a type of the NRAW come out. */
struct exported_type {
	std::shared_ptr<arrow::DataType> type;
	array_of_t array_of;
};

/**
 * The values of @p rows of @p column, which @p convert turns into values of @p type, or into
 * nulls.
 */
template <typename ArrowType, typename Value, typename Convert>
std::shared_ptr<arrow::Array> values_of(const pvcop::db::array& column,
                                        const rows_t& rows,
                                        const std::shared_ptr<arrow::DataType>& type,
                                        const Convert& convert)
{
	typename arrow::TypeTraits<ArrowType>::BuilderType builder(type, arrow::default_memory_pool());
	PVRush::PVParquetExporterBase::check(builder.Reserve(int64_t(rows.size())));
	const pvcop::core::array<Value>& values = column.to_core_array<Value>();
	const bool has_invalid = column.has_invalid();
	for (PVRow row : rows) {
		const std::optional<typename ArrowType::c_type> value =
		    has_invalid and not column.is_valid(row) ? std::nullopt : convert(values[row]);
		if (value) {
			builder.UnsafeAppend(*value);
		} else {
			builder.UnsafeAppendNull();
		}
	}
	return PVRush::PVParquetExporterBase::value_of(builder.Finish());
}

template <typename ArrowType, typename Value, typename Convert>
exported_type converted(const std::shared_ptr<arrow::DataType>& type, Convert convert)
{
	return {type, [type, convert](const pvcop::db::array& column, const rows_t& rows) {
		        return values_of<ArrowType, Value>(column, rows, type, convert);
	        }};
}

template <typename ArrowType, typename Value>
exported_type number()
{
	return converted<ArrowType, Value>(
	    arrow::TypeTraits<ArrowType>::type_singleton(),
	    [](Value value) { return std::optional<typename ArrowType::c_type>(value); });
}

/** The milliseconds since the epoch of a time in seconds since then. */
std::optional<int64_t> from_seconds(uint64_t seconds)
{
	return int64_t(seconds) * 1000;
}

std::optional<int64_t> from_milliseconds(uint64_t milliseconds)
{
	return int64_t(milliseconds);
}

/** The microseconds since the epoch of @p time, which a special value has none of. */
std::optional<int64_t> from_time(const boost::posix_time::ptime& time)
{
	static const boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
	if (time.is_special()) {
		return std::nullopt;
	}
	return (time - epoch).total_microseconds();
}

/** The microseconds of @p duration, which a special value has none of. */
std::optional<int64_t> from_duration(const boost::posix_time::time_duration& duration)
{
	if (duration.is_special()) {
		return std::nullopt;
	}
	return duration.total_microseconds();
}

/**
 * The types of the NRAW that keep a type of their own, by the names of their formatters.
 */
const std::unordered_map<std::string, exported_type>& exported_types()
{
	using boost::posix_time::ptime;
	using boost::posix_time::time_duration;
	static const std::shared_ptr<arrow::DataType> milliseconds =
	    arrow::timestamp(arrow::TimeUnit::MILLI);
	static const std::shared_ptr<arrow::DataType> microseconds =
	    arrow::timestamp(arrow::TimeUnit::MICRO);

	// Without a time zone: Squey reads a time written without one as UTC, and shows it so.
	static const std::unordered_map<std::string, exported_type> types = {
	    {"number_bool", number<arrow::BooleanType, bool>()},
	    {"number_int8", number<arrow::Int8Type, int8_t>()},
	    {"number_int16", number<arrow::Int16Type, int16_t>()},
	    {"number_int32", number<arrow::Int32Type, int32_t>()},
	    {"number_int64", number<arrow::Int64Type, int64_t>()},
	    {"number_uint8", number<arrow::UInt8Type, uint8_t>()},
	    {"number_uint16", number<arrow::UInt16Type, uint16_t>()},
	    {"number_uint32", number<arrow::UInt32Type, uint32_t>()},
	    {"number_uint64", number<arrow::UInt64Type, uint64_t>()},
	    {"number_float", number<arrow::FloatType, float>()},
	    {"number_double", number<arrow::DoubleType, double>()},
	    // Seconds, which Parquet has no unit for: readers would get milliseconds back anyway.
	    {"datetime", converted<arrow::TimestampType, uint64_t>(milliseconds, from_seconds)},
	    {"datetime_ms", converted<arrow::TimestampType, uint64_t>(milliseconds, from_milliseconds)},
	    {"datetime_us", converted<arrow::TimestampType, ptime>(microseconds, from_time)},
	    {"duration", converted<arrow::DurationType, time_duration>(
	                     arrow::duration(arrow::TimeUnit::MICRO), from_duration)},
	};
	return types;
}

/**
 * The text of @p rows of @p column, as the CSV export writes it.
 */
std::shared_ptr<arrow::Array> text_of(const pvcop::db::array& column, const rows_t& rows)
{
	arrow::StringBuilder builder;
	PVRush::PVParquetExporterBase::check(builder.Reserve(int64_t(rows.size())));
	// The strings of the dictionary, rather than a copy of each of them, but for invalid values.
	const pvcop::db::read_dict* dict = column.dict();
	const pvcop::core::array<pvcop::string_index_t>* indexes =
	    dict != nullptr ? &column.to_core_array<pvcop::string_index_t>() : nullptr;
	for (PVRow row : rows) {
		if (indexes != nullptr and column.is_valid(row)) {
			PVRush::PVParquetExporterBase::check(
			    builder.Append(std::string_view(dict->key((*indexes)[row]))));
		} else {
			PVRush::PVParquetExporterBase::check(builder.Append(column.at(row)));
		}
	}
	return PVRush::PVParquetExporterBase::value_of(builder.Finish());
}

/**
 * How @p column comes out: with a type of its own, or as text.
 */
exported_type exported_type_of(const pvcop::db::array& column)
{
	const auto& types = exported_types();
	const auto type = types.find(column.type());
	if (type != types.end()) {
		return type->second;
	}
	return {arrow::utf8(), text_of};
}

/**
 * @p names, those met before getting a number.
 */
std::vector<std::string> unique_names(const std::vector<std::string>& names)
{
	std::unordered_set<std::string> taken;
	std::vector<std::string> unique;
	for (const std::string& name : names) {
		std::string unique_name = name;
		for (int number = 2; not taken.insert(unique_name).second; number++) {
			unique_name = name + "_" + std::to_string(number);
		}
		unique.push_back(std::move(unique_name));
	}
	return unique;
}

} // namespace

PVRush::PVParquetNrawExporter::PVParquetNrawExporter(PVRush::PVNraw const& nraw,
                                                     QStringList column_names)
    : _nraw(nraw), _column_names(std::move(column_names))
{
}

void PVRush::PVParquetNrawExporter::export_rows(const std::string& path,
                                                const PVCore::PVSelBitField& sel)
{
	PVCore::PVColumnIndexes columns = _columns;
	if (columns.empty()) {
		for (PVCol column(0); column < _nraw.column_count(); column++) {
			columns.push_back(column);
		}
	}

	std::vector<std::string> names;
	std::vector<exported_type> types;
	for (PVCol column : columns) {
		if (column >= _nraw.column_count() or int(column) >= _column_names.size()) {
			throw PVExportError("The source has no column " + std::to_string(int(column)));
		}
		names.push_back(_column_names[int(column)].toStdString());
		types.push_back(exported_type_of(_nraw.column(column)));
	}
	names = unique_names(names);

	arrow::FieldVector fields;
	for (size_t i = 0; i < columns.size(); i++) {
		fields.push_back(arrow::field(names[i], types[i].type));
	}
	const std::shared_ptr<arrow::Schema> schema = arrow::schema(fields);

	std::unique_ptr<parquet::arrow::FileWriter> writer = open(path, schema);

	const PVRow row_count = std::min<PVRow>(_nraw.row_count(), sel.count());
	size_t exported_row_count = 0;
	rows_t rows;
	for (PVRow begin = 0; begin < row_count; begin += BATCH_ROW_COUNT) {
		const PVRow end = std::min<PVRow>(begin + BATCH_ROW_COUNT, row_count);
		sel.visit_selected_lines([&](PVRow row) { rows.push_back(row); }, end, begin);
		if (rows.empty() or (rows.size() < BATCH_ROW_COUNT and end < row_count)) {
			continue;
		}

		arrow::ArrayVector arrays(columns.size());
		tbb::parallel_for(size_t(0), columns.size(), [&](size_t i) {
			arrays[i] = types[i].array_of(_nraw.column(columns[i]), rows);
		});
		check(writer->WriteRecordBatch(
		    *arrow::RecordBatch::Make(schema, int64_t(rows.size()), std::move(arrays))));

		exported_row_count += rows.size();
		rows.clear();
		if (_f_progress) {
			_f_progress(exported_row_count);
		}
		if (close_if_canceled(*writer, path)) {
			return;
		}
	}

	check(writer->Close());
}
