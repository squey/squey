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

#include <arrow/api.h>
#include <arrow/array/array_base.h>
#include <arrow/array/array_binary.h>
#include <arrow/array/array_decimal.h>
#include <arrow/array/array_dict.h>
#include <arrow/array/array_primitive.h>
#include <arrow/array/data.h>
#include <arrow/type_traits.h>
#include <arrow/util/decimal.h>
#include <arrow/util/float16.h>
#include <qbytearray.h>
#include <qbytearrayview.h>
#include <qstring.h>
#include <stdint.h>
#include <time.h>
#include <boost/algorithm/string/replace.hpp>
#include <boost/date_time/posix_time/conversion.hpp>
#include <boost/date_time/posix_time/posix_time_config.hpp>
#include <boost/date_time/posix_time/posix_time_duration.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/date_time/time.hpp>
#include <algorithm>
#include <QCryptographicHash>
#include <bitset>
#include <cstring>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>

#include "parquet/PVParquetAPI.h"
#include "pvbase/types.h"
#include "pvcop/db/write_dict.h"
#include "pvkernel/core/PVBinaryChunk.h"

void* convert_bool(const std::shared_ptr<arrow::Array>& column_array, void* data, pvcop::db::write_dict* dict)
{
	const auto& bool_array = static_cast<arrow::BooleanArray&>(*column_array);

	dict->insert("false");
	dict->insert("true");

	std::transform(bool_array.begin(), bool_array.end(), (pvcop::db::index_t*)data, [&](const std::optional<int64_t>& value) { return value.value_or(0); });

	return data;
}

// StringArrayType is either arrow::StringArray (32 bits offsets) or arrow::LargeStringArray (64 bits offsets)
template <typename StringArrayType>
void* convert_string(const std::shared_ptr<arrow::Array>& column_array, pvcop::db::index_t* values, pvcop::db::write_dict* dict)
{
    auto& str_array = static_cast<StringArrayType&>(*column_array);

    for (int64_t i = 0; i < str_array.length(); ++i) {
        if (str_array.IsNull(i)) {
            values[i] = dict->insert("");
        } else {
            std::string value = str_array.GetString(i);
            boost::replace_all(value, "\n", "\\n"); // escape end of lines
            values[i] = dict->insert(value.c_str());
        }
    }

	return values;
}

template <typename StringArrayType>
static std::vector<pvcop::db::index_t> map_dictionary_values(const arrow::Array& dictionary, pvcop::db::write_dict* dict)
{
	const auto& values = static_cast<const StringArrayType&>(dictionary);

	std::vector<pvcop::db::index_t> dict_index_map;
	dict_index_map.reserve(values.length());

	std::transform(values.begin(), values.end(), std::back_inserter(dict_index_map),
		[&dict](auto str)
		{
		    std::string safe_str(std::string(str.value_or("")));
			boost::replace_all(safe_str, "\n", "\\n"); // escape end of lines
			return dict->insert(safe_str.c_str());
		}
	);

	return dict_index_map;
}

void* convert_dictionnary(const std::shared_ptr<arrow::Array>& column_array, pvcop::db::write_dict* dict)
{
	auto& dict_array = static_cast<arrow::DictionaryArray&>(*column_array);
	const std::shared_ptr<arrow::Array>& dictionary = dict_array.dictionary();
	auto& indices = static_cast<arrow::Int32Array&>(*dict_array.indices());

	const std::vector<pvcop::db::index_t> dict_index_map =
		dictionary->type_id() == arrow::Type::type::LARGE_STRING
			? map_dictionary_values<arrow::LargeStringArray>(*dictionary, dict)
			: map_dictionary_values<arrow::StringArray>(*dictionary, dict);

	// Arrow leaves the index of a null entry unspecified, and an all-null column comes with
	// an empty dictionary : both must be mapped to the empty string, as convert_string does,
	// instead of being used to index dict_index_map out of its bounds.
	std::optional<pvcop::db::index_t> null_index;
	auto index_of_null = [&]() {
		if (not null_index) {
			null_index = dict->insert("");
		}
		return *null_index;
	};

	std::span<int32_t> rw_indices{(int32_t*)indices.raw_values(), (size_t)indices.length()};
	for (int64_t i = 0; i < indices.length(); i++) {
		const int32_t index = rw_indices[i];
		const bool usable = not indices.IsNull(i) and index >= 0 and
		                    (size_t)index < dict_index_map.size();
		rw_indices[i] = usable ? dict_index_map[index] : index_of_null();
	}

	return ((void*)indices.raw_values());
}

template <typename T>
void* convert_binary(const std::shared_ptr<T>& binary_array, pvcop::db::index_t* values, pvcop::db::write_dict* dict)
{
	for (int64_t i = 0; i < binary_array->length(); ++i) {
		const auto& value = binary_array->GetString(i);
		QCryptographicHash hash(QCryptographicHash::Sha256);
        hash.addData(QByteArrayView(value.data(), value.size()));
        QString checksum = QString(hash.result().toHex()) + " (sha256)";
		values[i] = dict->insert(checksum.toStdString().c_str());
	}

	return values;
}

void* convert_half_float(const std::shared_ptr<arrow::Array>& column_array, void* data)
{
	const auto& half_floats = static_cast<arrow::HalfFloatArray&>(*column_array);
	float* floats = static_cast<float*>(data);

	// Widening a half float to a float is always lossless
	std::transform(half_floats.begin(), half_floats.end(), floats, [](const std::optional<uint16_t>& bits) {
		return arrow::util::Float16::FromBits(bits.value_or(0)).ToFloat();
	});

	return data;
}

// DecimalType is one of arrow::Decimal[32|64|128|256]Type
template <typename DecimalType>
void* convert_decimal(const std::shared_ptr<arrow::Array>& column_array, void* data)
{
	using decimal_array_t = typename arrow::TypeTraits<DecimalType>::ArrayType;
	using decimal_value_t = typename arrow::TypeTraits<DecimalType>::CType;

	const auto& decimals = static_cast<decimal_array_t&>(*column_array);
	const auto& type = static_cast<const arrow::DecimalType&>(*decimals.type());
	const int32_t scale = type.scale();
	double* doubles = static_cast<double*>(data);

	for (int64_t i = 0; i < decimals.length(); ++i) {
		// Decimals whose precision exceeds the 15 significant digits of a double are rounded
		doubles[i] = decimals.IsNull(i) ? 0. : decimal_value_t(decimals.GetValue(i)).ToDouble(scale);
	}

	return data;
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
    uint64_t* times = static_cast<uint64_t*>(data);

    std::transform(dates.begin(), dates.end(), times,
        [](const std::optional<int32_t>& days_since_epoch) {
            int32_t days = days_since_epoch.value_or(0);
            return static_cast<uint64_t>(days) * 86400;
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

void* convert_duration(const std::shared_ptr<arrow::Array>& column_array, void* data)
{
	auto& duration_array = static_cast<arrow::DurationArray&>(*column_array);
	const auto& type = static_cast<const arrow::DurationType&>(*duration_array.type());
	boost::posix_time::time_duration* durations = (boost::posix_time::time_duration*)data;
	switch (type.unit()) {
		case arrow::TimeUnit::SECOND: {
			std::transform(duration_array.begin(), duration_array.end(), durations, [](const std::optional<int64_t>& sec) {
				return boost::posix_time::seconds(sec.value_or(0));
			});
			break;
		}
		case arrow::TimeUnit::MILLI: {
			std::transform(duration_array.begin(), duration_array.end(), durations, [](const std::optional<int64_t>& ms) {
				return boost::posix_time::millisec(ms.value_or(0));
			});
			break;
		}
		case arrow::TimeUnit::MICRO: {
			std::transform(duration_array.begin(), duration_array.end(), durations, [](const std::optional<int64_t>& us) {
				return boost::posix_time::microsec(us.value_or(0));
			});
			break;
		}
		case arrow::TimeUnit::NANO: {
			// a boost time duration has a microsecond resolution
			std::transform(duration_array.begin(), duration_array.end(), durations, [](const std::optional<int64_t>& ns) {
				return boost::posix_time::microsec(ns.value_or(0) / 1000);
			});
			break;
		}
	}
	return data;
}

// ListArrayType is one of arrow::[Large]ListArray or arrow::FixedSizeListArray
template <typename ListArrayType>
static void list_to_string(const arrow::Array& column_array, int64_t row, std::stringstream& ss)
{
	const auto& list_array = static_cast<const ListArrayType&>(column_array);
	const int64_t start = list_array.value_offset(row);
	const int64_t end = start + list_array.value_length(row);
	const std::shared_ptr<arrow::Array>& values_array = list_array.values();

	ss << "[";
	for (int64_t j = start; j < end; ++j) {
		if (j != start) ss << ", ";
		ss << values_array->GetScalar(j).ValueOrDie()->ToString();
	}
	ss << "]";
}

void* convert_complex_type_as_string(
    const std::shared_ptr<arrow::Array>& column_array,
    pvcop::db::index_t* values,
    pvcop::db::write_dict* dict)
{
    std::stringstream ss;

    for (int64_t i = 0; i < column_array->length(); ++i) {
        if (column_array->IsNull(i)) {
            values[i] = dict->insert(""); // empty string for null
            continue;
        }

        ss.str("");
        ss.clear();

        switch (column_array->type_id()) {

            case arrow::Type::LIST:
                list_to_string<arrow::ListArray>(*column_array, i, ss);
                break;

            case arrow::Type::LARGE_LIST:
                list_to_string<arrow::LargeListArray>(*column_array, i, ss);
                break;

            case arrow::Type::FIXED_SIZE_LIST:
                list_to_string<arrow::FixedSizeListArray>(*column_array, i, ss);
                break;

            case arrow::Type::MAP: {
                auto& map_array = static_cast<arrow::MapArray&>(*column_array);
                auto start = map_array.value_offset(i);
                auto end = map_array.value_offset(i + 1);
                auto key_array = map_array.keys();
                auto item_array = map_array.items();

                ss << "{";
                for (int64_t j = start; j < end; ++j) {
                    if (j != start) ss << ",";
                    auto key = key_array->GetScalar(j).ValueOrDie();
                    auto val = item_array->GetScalar(j).ValueOrDie();

                    if (key_array->type()->id() == arrow::Type::STRING) {
                        ss << "\"" << key->ToString() << "\":" << val->ToString();
                    } else {
                        ss << key->ToString() << ":" << val->ToString();
                    }
                }
                ss << "}";
                break;
            }

            default:
                // fallback: treat as string
                ss << column_array->GetScalar(i).ValueOrDie()->ToString();
                break;
        }

        values[i] = dict->insert(ss.str().c_str());
    }

    return values;
}

PVRush::PVParquetBinaryChunk::PVParquetBinaryChunk(
    bool multi_inputs,
    bool is_bit_optimizable,
    size_t input_index,
    std::shared_ptr<arrow::Table>& table,
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
			const auto& column = table->column(column_index);
			const std::shared_ptr<arrow::Array>& column_array = column->chunk(0);

			if (column_array == nullptr) {
				continue;
			}

			const arrow::Type::type type_id = column_array->type_id();
			const auto& t = PVParquetAPI::pvcop_types_map.at(type_id);
			_values[i].reserve(row_count * t.size_in_bytes);

			// Note : the values buffer must not be accessed before knowing the type, as some arrays
			// (FIXED_SIZE_LIST) only hold a validity bitmap.
			void* data = nullptr;
			switch (type_id) {
				case arrow::Type::type::BOOL:
					data = convert_bool(column_array, _values[i].data(), dicts[i]);
					break;
				case arrow::Type::type::STRING:
					data = convert_string<arrow::StringArray>(column_array, (pvcop::db::index_t*)_values[i].data(), dicts[i]);
					break;
				case arrow::Type::type::LARGE_STRING:
					data = convert_string<arrow::LargeStringArray>(column_array, (pvcop::db::index_t*)_values[i].data(), dicts[i]);
					break;
				case arrow::Type::type::STRING_VIEW:
					data = convert_string<arrow::StringViewArray>(column_array, (pvcop::db::index_t*)_values[i].data(), dicts[i]);
					break;
				case arrow::Type::type::DICTIONARY:
					data = convert_dictionnary(column_array, dicts[i]);
					break;
				case arrow::Type::type::FIXED_SIZE_BINARY:
					data = convert_binary(std::static_pointer_cast<arrow::FixedSizeBinaryArray>(column_array), (pvcop::db::index_t*)_values[i].data(), dicts[i]);
					break;
				case arrow::Type::type::BINARY:
					data = convert_binary(std::static_pointer_cast<arrow::BinaryArray>(column_array), (pvcop::db::index_t*)_values[i].data(), dicts[i]);
					break;
				case arrow::Type::type::LARGE_BINARY:
					data = convert_binary(std::static_pointer_cast<arrow::LargeBinaryArray>(column_array), (pvcop::db::index_t*)_values[i].data(), dicts[i]);
					break;
				case arrow::Type::type::BINARY_VIEW:
					data = convert_binary(std::static_pointer_cast<arrow::BinaryViewArray>(column_array), (pvcop::db::index_t*)_values[i].data(), dicts[i]);
					break;
				case arrow::Type::type::HALF_FLOAT:
					data = convert_half_float(column_array, _values[i].data());
					break;
				case arrow::Type::type::DECIMAL32:
					data = convert_decimal<arrow::Decimal32Type>(column_array, _values[i].data());
					break;
				case arrow::Type::type::DECIMAL64:
					data = convert_decimal<arrow::Decimal64Type>(column_array, _values[i].data());
					break;
				case arrow::Type::type::DECIMAL128:
					data = convert_decimal<arrow::Decimal128Type>(column_array, _values[i].data());
					break;
				case arrow::Type::type::DECIMAL256:
					data = convert_decimal<arrow::Decimal256Type>(column_array, _values[i].data());
					break;
				case arrow::Type::type::TIMESTAMP:
					data = convert_timestamp(column_array, _values[i].data());
					break;
				case arrow::Type::type::DATE32:
					data = convert_date32(column_array, _values[i].data());
					break;
				case arrow::Type::type::TIME32:
					data = convert_time32(column_array, _values[i].data());
					break;
				case arrow::Type::type::TIME64:
					data = convert_time64(column_array, _values[i].data());
					break;
				case arrow::Type::type::DURATION:
					data = convert_duration(column_array, _values[i].data());
					break;
				case arrow::Type::type::LIST:
				case arrow::Type::type::LARGE_LIST:
				case arrow::Type::type::FIXED_SIZE_LIST:
				case arrow::Type::type::MAP:
					data = convert_complex_type_as_string(column_array, (pvcop::db::index_t*)_values[i].data(), dicts[i]);
					break;
				default: // fixed-width primitive types are copied as-is below
					data = ((void*)column_array->data()->buffers[1]->data());
					break;
			}

			// handle null values (optimized)
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
    			const uint8_t* base_ptr = static_cast<const uint8_t*>(column_array->data()->buffers[1]->data());
                const uint8_t* src_ptr = base_ptr + column_array->offset() * t.size_in_bytes;
                std::memcpy(_values[i].data(), src_ptr, row_count * t.size_in_bytes);
			}
			set_raw_column_chunk(PVCol(i+multi_inputs), _values[i].data(), row_count, t.size_in_bytes, t.string);
		}

		// handle null values (not optimized)
		if (not is_bit_optimizable) {
			for (size_t i = 0 ; i < column_indexes.size(); i++) {
				const size_t col = column_indexes[i];
				const std::shared_ptr<arrow::Array>& column_array = table->column(col)->chunk(0);
				if (column_array->null_count() > 0) {
					set_invalid_column(PVCol(i+multi_inputs));
					for (PVRow row = 0; row < column_array->length(); ++row) {
						if (column_array->IsNull(row)) {
							set_invalid(PVCol(i+multi_inputs), row);
						}
					}
				}
			}
		}
	}
