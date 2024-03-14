/* * MIT License
 *
 * © Squey, 2024
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

#include "PVParquetBinaryChunk.h"

#include <execution>

#include "boost/date_time/posix_time/posix_time.hpp"

#include <omp.h>

void* convert_bool(const std::shared_ptr<arrow::Array>& column_array, void* data, pvcop::db::write_dict* dict)
{
	const auto& bool_array = static_cast<arrow::BooleanArray&>(*column_array);

	dict->insert("false");
	dict->insert("true");

	std::transform(bool_array.begin(), bool_array.end(), (pvcop::db::index_t*)data, [&](const std::optional<int64_t>& value) { return value.value_or(0); });

	return data;
}

void* convert_dictionnary(const std::shared_ptr<arrow::Array>& column_array, pvcop::db::write_dict* dict)
{
	auto& dict_array = static_cast<arrow::DictionaryArray&>(*column_array);
	const auto& dictionary = static_cast<arrow::StringArray&>(*dict_array.dictionary());
	auto& indices = static_cast<arrow::Int32Array&>(*dict_array.indices());

	std::vector<pvcop::db::index_t> dict_index_map;
	dict_index_map.reserve(dictionary.length());

	std::transform(dictionary.begin(), dictionary.end(), std::back_inserter(dict_index_map),
		[&dict](auto str){ return dict->insert(std::string(str.value_or("")).c_str()); });
	std::span<int32_t> rw_indices{(int32_t*)indices.raw_values(), (size_t)indices.length()};
	std::for_each(rw_indices.begin(), rw_indices.end(), [&dict_index_map](int32_t& index) { index = dict_index_map[index]; });

	return ((void*)indices.raw_values());
}

template <typename T>
void* convert_binary(const std::shared_ptr<T>& binary_array, pvcop::db::index_t* values, pvcop::db::write_dict* dict)
{
	for (int64_t i = 0; i < binary_array->length(); ++i) {
		std::ostringstream oss;
		oss << "b'";
		const auto& value = binary_array->GetString(i);
		const uint8_t* bytes = reinterpret_cast<const uint8_t*>(value.c_str());
		for (size_t j = 0; j < value.size(); ++j) {
			oss << std::hex << static_cast<uint8_t>(bytes[j]);
		}
		oss << "'";
		values[i] = dict->insert(oss.str().c_str());
	}

	return values;
}

void* convert_timestamp(const std::shared_ptr<arrow::Array>& column_array, void* data)
{
	auto& timestamps = static_cast<arrow::TimestampArray&>(*column_array);
	const auto& type = static_cast<const arrow::TimestampType&>(*timestamps.type());
	boost::posix_time::ptime* ptimes = (boost::posix_time::ptime*)data;
	switch (type.unit()) {
		case arrow::TimeUnit::MILLI: {
			std::transform(timestamps.begin(), timestamps.end(), ptimes, [](const std::optional<int64_t>& time) {
				return boost::posix_time::from_time_t(0) + boost::posix_time::milliseconds(time.value_or(0));
			});
			break;
		}
		case arrow::TimeUnit::MICRO: {
			std::transform(timestamps.begin(), timestamps.end(), ptimes, [](const std::optional<int64_t>& time) {
				return boost::posix_time::from_time_t(0) + boost::posix_time::microseconds(time.value_or(0));
			});
			break;
		}
		case arrow::TimeUnit::NANO: {
			std::transform(timestamps.begin(), timestamps.end(), ptimes, [](const std::optional<int64_t>& time) {
				return boost::posix_time::from_time_t(0) + boost::posix_time::microseconds(time.value_or(0) / 1000);
			});
			break;
		}
		case arrow::TimeUnit::SECOND:
		default:
			break;
	}
	return data;
}

void* convert_date32(const std::shared_ptr<arrow::Array>& column_array, void* data)
{
	auto& dates = static_cast<arrow::Date32Array&>(*column_array);
	uint64_t* times = (uint64_t*)data;
	std::transform(dates.begin(), dates.end(), times, [&](const std::optional<int32_t>& days_since_epoch) {
		tm date = {};
		date.tm_year = 70;
		date.tm_isdst = -1;
		date.tm_mday = days_since_epoch.value_or(0)+1;
		return mktime(&date);
	});
	return data;
}

void* convert_time32(const std::shared_ptr<arrow::Array>& column_array, void* data)
{
	auto& time32 = static_cast<arrow::Time32Array&>(*column_array);
	const auto& type = static_cast<const arrow::Time32Type&>(*time32.type());
	boost::posix_time::time_duration* durations = (boost::posix_time::time_duration*)data;
	if (type.unit() == arrow::TimeUnit::SECOND) {
		std::transform(time32.begin(), time32.end(), durations, [&](const std::optional<int32_t>& sec) {
			return boost::posix_time::seconds(sec.value_or(0));
		});
	}
	else { // type.unit() == arrow::TimeUnit::MILLI
		std::transform(time32.begin(), time32.end(), durations, [&](const std::optional<int32_t>& ms) {
			return boost::posix_time::millisec(ms.value_or(0));
		});
	}
	return data;
}

void* convert_time64(const std::shared_ptr<arrow::Array>& column_array, void* data)
{
	auto& time64 = static_cast<arrow::Time64Array&>(*column_array);
	const auto& type = static_cast<const arrow::Time64Type&>(*time64.type());
	boost::posix_time::time_duration* durations = (boost::posix_time::time_duration*)data;
	if (type.unit() == arrow::TimeUnit::MICRO) {
		std::transform(time64.begin(), time64.end(), durations, [&](const std::optional<int64_t>& us) {
			return boost::posix_time::microsec(us.value_or(0));
		});
	}
	else { // type.unit() == arrow::TimeUnit::NANO
		std::transform(time64.begin(), time64.end(), durations, [&](const std::optional<int64_t>& ns) {
			return boost::posix_time::microsec(ns.value_or(0) / 1000);
		});
	}
	return data;
}

PVRush::PVParquetBinaryChunk::PVParquetBinaryChunk(
    bool multi_inputs,
    bool is_bit_optimizable,
    size_t input_index,
    arrow::RecordBatch* record_batch,
    const std::vector<size_t>& column_indexes,
    std::vector<pvcop::db::write_dict*>& dicts,
    size_t row_count,
    size_t nraw_start_row
    )
	: PVCore::PVBinaryChunk(column_indexes.size() + multi_inputs, row_count, (PVRow)nraw_start_row)
	{
		set_init_size(row_count * MEGA);
		_values.resize(column_indexes.size());

		if (multi_inputs) {
			_input_index = std::vector<pvcop::db::index_t>(row_count, (pvcop::db::index_t)input_index);
			set_raw_column_chunk(PVCol(0), (void*)(_input_index.data()), row_count, sizeof(pvcop::db::index_t), "string");
		}
		
#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0 ; i < column_indexes.size(); i++) {
			const size_t column_index = column_indexes[i];
			const std::shared_ptr<arrow::Array>& column_array = record_batch->column(column_index);

			if (column_array == nullptr) {
				continue;
			}

			const arrow::Type::type type_id = column_array->type_id();
			const auto& t = PVParquetAPI::pvcop_types_map.at(type_id);
			_values[i].reserve(row_count * t.size_in_bytes);

			void* data = ((void*)column_array->data()->buffers[1]->data());
			if (column_array->type_id() == arrow::Type::type::BOOL) {
				data = convert_bool(column_array, _values[i].data(), dicts[i]);
			}
			if (column_array->type_id() == arrow::Type::type::DICTIONARY) {
				data = convert_dictionnary(column_array, dicts[i]);
			}
			else if (column_array->type_id() == arrow::Type::type::FIXED_SIZE_BINARY) {
				const auto& binary_array = std::static_pointer_cast<arrow::FixedSizeBinaryArray>(column_array);
				data = convert_binary(binary_array, (pvcop::db::index_t*)_values[i].data(), dicts[i]);
			}
			else if (column_array->type_id() == arrow::Type::type::BINARY) {
				const auto& binary_array = std::static_pointer_cast<arrow::BinaryArray>(column_array);
				data = convert_binary(binary_array, (pvcop::db::index_t*)_values[i].data(), dicts[i]);
			}
            else if (column_array->type_id() == arrow::Type::type::TIMESTAMP) {
				data = convert_timestamp(column_array, _values[i].data());
			}
			else if (column_array->type_id() == arrow::Type::type::DATE32) {
				data = convert_date32(column_array, _values[i].data());
			}
			else if (column_array->type_id() == arrow::Type::type::TIME32) {
				data = convert_time32(column_array, _values[i].data());
			}
			else if (column_array->type_id() == arrow::Type::type::TIME64) {
				data = convert_time64(column_array, _values[i].data());
			}

			// handle null values for single input (optimized)
			if (is_bit_optimizable and column_array->null_count() > 0) {
				const uint8_t* null_bitmap_data = column_array->null_bitmap_data();
				constexpr const int digits = std::numeric_limits<uint8_t>::digits;
				size_t null_bitmap_data_size = (row_count + digits - 1) / digits;
				auto null_bitmap_data_ptr = std::make_unique<uint8_t[]>(null_bitmap_data_size);
				std::memcpy(null_bitmap_data_ptr.get(), null_bitmap_data, null_bitmap_data_size);
				std::for_each(null_bitmap_data_ptr.get(), null_bitmap_data_ptr.get() + null_bitmap_data_size, [](uint8_t& byte) {
					byte = static_cast<uint8_t>(std::bitset<8>(byte).flip().to_ulong());
				});
				set_null_bitmap(PVCol(i+multi_inputs), std::move(null_bitmap_data_ptr));
			}

			if (_values[i].data() != data) {
				std::memcpy(_values[i].data(), data, _values[i].capacity());
			}
			set_raw_column_chunk(PVCol(i+multi_inputs), _values[i].data(), row_count, t.size_in_bytes, t.string);
		}

		// handle null values for multi inputs (not optimized)
		if (not is_bit_optimizable) {
			for (size_t i = 0 ; i < column_indexes.size(); i++) {
				const size_t column_index = column_indexes[i];
				const std::shared_ptr<arrow::Array>& column_array = record_batch->column(column_index);
				if (column_array->null_count() > 0) {
					set_invalid_column(PVCol(i+1));
					for (PVRow j = 0; j < column_array->length(); ++j) {
						if (column_array->IsNull(j)) {
							set_invalid(PVCol(i+1), j);
						}
					}
				}
			}
		}
	}