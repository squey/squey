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

/**
 * Check the support of the LARGE_STRING, LARGE_BINARY, LARGE_LIST, FIXED_SIZE_LIST, STRING_VIEW,
 * BINARY_VIEW, HALF_FLOAT, DECIMAL[32|64|128|256] and DURATION parquet types.
 *
 * Values are checked directly against the NRaw content rather than against a CSV reference file,
 * as the textual representation of floating point numbers differs between platforms.
 */

#include <memory>

#include "../../plugins/common/parquet/PVParquetAPI.h"
#include "../../plugins/common/parquet/PVParquetFileDescription.h"
#include "../../plugins/input-types/parquet/PVParquetExporter.h"
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/core/squey_assert.h>

#include "common.h"

#include <pvlogger.h>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/util/decimal.h>
#include <arrow/util/float16.h>

#include <parquet/arrow/writer.h>

#include <boost/algorithm/string/replace.hpp>

#include <cmath>
#include <optional>
#include <string>
#include <vector>

/**
 * The values below describe PATTERN_SIZE rows, repeated up to ROW_COUNT. Spanning more than the
 * 64k rows of a record batch makes sure the columns are properly rebuilt across several chunks,
 * the string dictionaries being shared by all of them.
 */
static constexpr const size_t PATTERN_SIZE = 8;
static constexpr const size_t ROW_COUNT = PATTERN_SIZE * 9000; // 72000 rows, ie. 2 record batches

/**
 * Rows holding a null value in every column.
 */
static bool is_null_row(size_t row)
{
	const size_t pattern_row = row % PATTERN_SIZE;

	return pattern_row == 2 or pattern_row == 5;
}

/**
 * The name of each column is the pvcop type it is expected to be mapped to,
 * the "#N" suffix being only there to make the names unique (see check_format).
 */
static std::shared_ptr<arrow::Schema> test_schema()
{
	return arrow::schema({
		arrow::field("string",          arrow::large_utf8()),
		arrow::field("string#1",        arrow::large_binary()),
		arrow::field("string#2",        arrow::large_list(arrow::int64())),
		arrow::field("string#3",        arrow::fixed_size_list(arrow::int64(), 2)),
		arrow::field("string#4",        arrow::utf8_view()),
		arrow::field("string#5",        arrow::binary_view()),
		arrow::field("number_float",    arrow::float16()),
		arrow::field("number_double",   arrow::decimal32(9, 2)),
		arrow::field("number_double#1", arrow::decimal64(18, 4)),
		arrow::field("number_double#2", arrow::decimal128(38, 10)),
		arrow::field("number_double#3", arrow::decimal256(50, 10)),
		arrow::field("duration",        arrow::duration(arrow::TimeUnit::SECOND)),
		arrow::field("duration#1",      arrow::duration(arrow::TimeUnit::MILLI)),
		arrow::field("duration#2",      arrow::duration(arrow::TimeUnit::MICRO)),
		arrow::field("duration#3",      arrow::duration(arrow::TimeUnit::NANO)),
	});
}

static const std::vector<std::optional<std::string>> large_strings = {
	"",
	"hello",
	std::nullopt,
	"multi\nline",           // end of lines are escaped by the parquet source
	"héllo wörld",           // multi-bytes characters exercise the byte offsets
	std::nullopt,
	"a longer string, to make sure 64 bits offsets are properly handled",
	"z",
};

/**
 * Values shared by the LARGE_BINARY and BINARY_VIEW columns. A view stores values of up to 12 bytes
 * inline and the longer ones in a separate buffer, so both sides of that boundary are covered.
 */
static const std::vector<std::optional<std::string>> binaries = {
	std::string(""),
	std::string("abc"),
	std::nullopt,
	std::string("\x00\x01\x02\x03", 4),               // embedded null bytes
	std::string("a binary value longer than 12 bytes"),
	std::nullopt,
	std::string("\xff\x00\xfe", 3),                   // non UTF-8 bytes
	std::string("z"),
};

/**
 * Binary values are replaced by the sha256 checksum of their content. Hardcoding the expected
 * checksums makes sure the whole content is hashed, null bytes included.
 */
static const std::vector<std::string> expected_checksums = {
	"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 (sha256)",
	"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad (sha256)",
	{},                                   // null
	"054edec1d0211f624fed0cbca9d4f9400b0e491c43742af2c5b0abebf0c990d8 (sha256)",
	"6be5ba3381fe823e02e00fca1de03c91dc071c88c180cb5c67649eb018de3ad8 (sha256)",
	{},                                   // null
	"af9ceddc9d8b08ac09e1994bfd20459b5e377425df7354dfce3501992828a5b7 (sha256)",
	"594e519ae499312b29433b7dd8a97ff068defcba9755b6d5d00e84c524d67b06 (sha256)",
};

/**
 * Same inline/out-of-line boundary as above, on the text side.
 */
static const std::vector<std::optional<std::string>> string_views = {
	"",
	"short",
	std::nullopt,
	"exactly12chr",                       // 12 bytes : biggest inline value
	"one byte too long to be inlined",
	std::nullopt,
	"héllo",                              // multi-bytes characters
	"z",
};

/**
 * Lists are rendered as strings. FIXED_SIZE_LIST holds a constant number of values per row,
 * unlike LARGE_LIST.
 */
static constexpr const int32_t FIXED_LIST_SIZE = 2;

static const std::vector<std::optional<std::vector<int64_t>>> large_lists = {
	std::vector<int64_t>{},                  // empty list
	std::vector<int64_t>{1, 2, 3},
	std::nullopt,
	std::vector<int64_t>{42},
	std::vector<int64_t>{-1, 0, 1},
	std::nullopt,
	std::vector<int64_t>{7, 8},
	std::vector<int64_t>{9},
};

static const std::vector<std::string> expected_large_lists = {
	"[]", "[1, 2, 3]", {}, "[42]", "[-1, 0, 1]", {}, "[7, 8]", "[9]",
};

/**
 * Arrow cannot write a FIXED_SIZE_LIST holding null values to parquet ("Lists with non-zero length
 * null components are not supported"), as a null entry still spans FIXED_LIST_SIZE slots. Such a
 * column therefore never holds any null value.
 */
static constexpr const int FIXED_SIZE_LIST_COLUMN = 3;

static const std::vector<std::optional<std::vector<int64_t>>> fixed_size_lists = {
	std::vector<int64_t>{0, 0},
	std::vector<int64_t>{1, 2},
	std::vector<int64_t>{11, 12},
	std::vector<int64_t>{3, 4},
	std::vector<int64_t>{-5, 6},
	std::vector<int64_t>{13, 14},
	std::vector<int64_t>{7, 8},
	std::vector<int64_t>{9, 10},
};

static const std::vector<std::string> expected_fixed_size_lists = {
	"[0, 0]", "[1, 2]", "[11, 12]", "[3, 4]", "[-5, 6]", "[13, 14]", "[7, 8]", "[9, 10]",
};

/**
 * Durations are stored as boost time durations, and rendered as "[-][D:]HH:MM:SS[.ffffff]".
 * Values are given in the time unit of their own column, and can be negative unlike a TIME32/TIME64.
 */
struct duration_values {
	std::vector<std::optional<int64_t>> values;
	std::vector<std::string> expected_strings;
};

static const duration_values duration_s_values = {
	{ 0, 1, std::nullopt, -61, 3661, std::nullopt, 90061, 86399 },
	{ "00:00:00", "00:00:01", {}, "-00:01:01", "01:01:01", {}, "1:01:01:01", "23:59:59" }
};

static const duration_values duration_ms_values = {
	{ 0, 1500, std::nullopt, -61000, 3661000, std::nullopt, 90061000, 86399999 },
	{ "00:00:00", "00:00:01.500000", {}, "-00:01:01", "01:01:01", {}, "1:01:01:01", "23:59:59.999000" }
};

static const duration_values duration_us_values = {
	{ 0, 1500000, std::nullopt, -61000000, 3661000000, std::nullopt, 90061000000, 1234567 },
	{ "00:00:00", "00:00:01.500000", {}, "-00:01:01", "01:01:01", {}, "1:01:01:01", "00:00:01.234567" }
};

// The last row makes the truncation of nanoseconds to the microsecond resolution explicit
static const duration_values duration_ns_values = {
	{ 0, 1500000000, std::nullopt, -61000000000, 3661000000000, std::nullopt, 90061000000000, 1234567890 },
	{ "00:00:00", "00:00:01.500000", {}, "-00:01:01", "01:01:01", {}, "1:01:01:01", "00:00:01.234567" }
};

/**
 * Half floats exactly representable as floats (which is always the case) and whose decimal
 * representation is short enough to be formatted the same way on every platform.
 */
static const std::vector<std::optional<float>> half_floats = {
	0.f,
	1.5f,
	std::nullopt,
	-2.25f,
	0.5f,
	std::nullopt,
	65504.f,                 // biggest finite half float
	-0.0625f,
};

/**
 * Decimals are converted to doubles. All the expected values below are exactly representable
 * as doubles, so that they can be compared without any tolerance.
 */
struct decimal_values {
	std::vector<std::optional<std::string>> values;
	std::vector<double> expected_doubles;
};

static const decimal_values decimal32_values = {
	{ "0.00", "1.25", std::nullopt, "-9.75", "100.50", std::nullopt, "9999999.75", "-0.25" },
	{ 0., 1.25, 0., -9.75, 100.5, 0., 9999999.75, -0.25 }
};

static const decimal_values decimal64_values = {
	{ "0.0000", "1.2500", std::nullopt, "-9.7500", "100.5000", std::nullopt, "99999999999999.9375", "-0.0625" },
	{ 0., 1.25, 0., -9.75, 100.5, 0., 99999999999999.9375, -0.0625 }
};

// 1208925819614629174706176 is 2^80 : its 2^-10 fractional part is way below the precision of a
// double, which makes the rounding of out-of-range decimals explicit.
static const decimal_values decimal128_values = {
	{ "0.0000000000", "1.2500000000", std::nullopt, "-9.7500000000", "100.5000000000", std::nullopt,
	  "1208925819614629174706176.0009765625", "-0.0009765625" },
	{ 0., 1.25, 0., -9.75, 100.5, 0., std::ldexp(1., 80), -0.0009765625 }
};

// 340282366920938463463374607431768211456 is 2^128, which doesn't fit in a decimal128
static const decimal_values decimal256_values = {
	{ "0.0000000000", "1.2500000000", std::nullopt, "-9.7500000000", "100.5000000000", std::nullopt,
	  "340282366920938463463374607431768211456.0009765625", "-0.0009765625" },
	{ 0., 1.25, 0., -9.75, 100.5, 0., std::ldexp(1., 128), -0.0009765625 }
};

/**
 * Repeat the values of a pattern up to ROW_COUNT rows.
 */
template <typename BuilderType, typename ValueType>
static std::shared_ptr<arrow::Array> make_array(BuilderType& builder,
                                                const std::vector<std::optional<ValueType>>& pattern)
{
	for (size_t row = 0; row < ROW_COUNT; row++) {
		const std::optional<ValueType>& value = pattern[row % PATTERN_SIZE];
		PARQUET_THROW_NOT_OK(value ? builder.Append(*value) : builder.AppendNull());
	}

	return builder.Finish().ValueOrDie();
}

static std::shared_ptr<arrow::Array> make_large_string_array()
{
	arrow::LargeStringBuilder builder;

	return make_array(builder, large_strings);
}

static std::shared_ptr<arrow::Array> make_string_view_array()
{
	arrow::StringViewBuilder builder;

	return make_array(builder, string_views);
}

static std::shared_ptr<arrow::Array> make_large_binary_array()
{
	arrow::LargeBinaryBuilder builder;

	return make_array(builder, binaries);
}

static std::shared_ptr<arrow::Array> make_binary_view_array()
{
	arrow::BinaryViewBuilder builder;

	return make_array(builder, binaries);
}

static std::shared_ptr<arrow::Array> make_duration_array(const std::shared_ptr<arrow::DataType>& type,
                                                         const duration_values& durations)
{
	arrow::DurationBuilder builder(type, arrow::default_memory_pool());

	return make_array(builder, durations.values);
}

// ListBuilderType is one of arrow::LargeListBuilder or arrow::FixedSizeListBuilder
template <typename ListBuilderType>
static std::shared_ptr<arrow::Array> make_list_array(
    ListBuilderType& builder,
    const std::shared_ptr<arrow::Int64Builder>& value_builder,
    const std::vector<std::optional<std::vector<int64_t>>>& lists)
{
	for (size_t row = 0; row < ROW_COUNT; row++) {
		const std::optional<std::vector<int64_t>>& list = lists[row % PATTERN_SIZE];
		if (not list) {
			PARQUET_THROW_NOT_OK(builder.AppendNull());
			continue;
		}
		PARQUET_THROW_NOT_OK(builder.Append());
		PARQUET_THROW_NOT_OK(value_builder->AppendValues(*list));
	}

	return builder.Finish().ValueOrDie();
}

static std::shared_ptr<arrow::Array> make_large_list_array()
{
	auto value_builder = std::make_shared<arrow::Int64Builder>();
	arrow::LargeListBuilder builder(arrow::default_memory_pool(), value_builder);

	return make_list_array(builder, value_builder, large_lists);
}

static std::shared_ptr<arrow::Array> make_fixed_size_list_array()
{
	auto value_builder = std::make_shared<arrow::Int64Builder>();
	arrow::FixedSizeListBuilder builder(arrow::default_memory_pool(), value_builder, FIXED_LIST_SIZE);

	return make_list_array(builder, value_builder, fixed_size_lists);
}

static std::shared_ptr<arrow::Array> make_half_float_array()
{
	arrow::HalfFloatBuilder builder;

	for (size_t row = 0; row < ROW_COUNT; row++) {
		const std::optional<float>& value = half_floats[row % PATTERN_SIZE];
		PARQUET_THROW_NOT_OK(value ? builder.Append(arrow::util::Float16::FromFloat(*value).bits())
		                           : builder.AppendNull());
	}

	return builder.Finish().ValueOrDie();
}

// DecimalType is one of arrow::Decimal[32|64|128|256]Type
template <typename DecimalType>
static std::shared_ptr<arrow::Array> make_decimal_array(const std::shared_ptr<arrow::DataType>& type,
                                                        const decimal_values& decimals)
{
	using builder_t = typename arrow::TypeTraits<DecimalType>::BuilderType;
	using decimal_t = typename arrow::TypeTraits<DecimalType>::CType;

	const int32_t column_scale = static_cast<const arrow::DecimalType&>(*type).scale();
	builder_t builder(type, arrow::default_memory_pool());

	for (size_t row = 0; row < ROW_COUNT; row++) {
		const std::optional<std::string>& value = decimals.values[row % PATTERN_SIZE];
		if (not value) {
			PARQUET_THROW_NOT_OK(builder.AppendNull());
			continue;
		}
		decimal_t decimal;
		int32_t precision;
		int32_t scale;
		PARQUET_THROW_NOT_OK(decimal_t::FromString(*value, &decimal, &precision, &scale));
		PARQUET_THROW_NOT_OK(builder.Append(decimal.Rescale(scale, column_scale).ValueOrDie()));
	}

	return builder.Finish().ValueOrDie();
}

static std::string generate_parquet_file(const std::string& file_path)
{
	const std::shared_ptr<arrow::Schema>& schema = test_schema();

	std::shared_ptr<arrow::RecordBatch> record_batch = arrow::RecordBatch::Make(schema, ROW_COUNT, {
		make_large_string_array(),
		make_large_binary_array(),
		make_large_list_array(),
		make_fixed_size_list_array(),
		make_string_view_array(),
		make_binary_view_array(),
		make_half_float_array(),
		make_decimal_array<arrow::Decimal32Type>(schema->field(7)->type(), decimal32_values),
		make_decimal_array<arrow::Decimal64Type>(schema->field(8)->type(), decimal64_values),
		make_decimal_array<arrow::Decimal128Type>(schema->field(9)->type(), decimal128_values),
		make_decimal_array<arrow::Decimal256Type>(schema->field(10)->type(), decimal256_values),
		make_duration_array(schema->field(11)->type(), duration_s_values),
		make_duration_array(schema->field(12)->type(), duration_ms_values),
		make_duration_array(schema->field(13)->type(), duration_us_values),
		make_duration_array(schema->field(14)->type(), duration_ns_values),
	});

	// The arrow schema must be stored in the parquet metadata, as parquet has no native
	// large string type and stores every decimal width as a single logical type.
	std::shared_ptr<parquet::ArrowWriterProperties> arrow_props =
	    parquet::ArrowWriterProperties::Builder().store_schema()->build();

	std::shared_ptr<arrow::io::FileOutputStream> output_file =
	    arrow::io::FileOutputStream::Open(file_path).ValueOrDie();
	std::unique_ptr<parquet::arrow::FileWriter> file_writer =
	    parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), output_file,
	                                     parquet::default_writer_properties(), arrow_props).ValueOrDie();
	std::shared_ptr<arrow::Table> table = arrow::Table::FromRecordBatches({record_batch}).ValueOrDie();
	PARQUET_THROW_NOT_OK(file_writer->WriteTable(*table));
	PARQUET_THROW_NOT_OK(file_writer->Close());

	return file_path;
}

static void import_file(
    const QString& file,
    QList<std::shared_ptr<PVRush::PVInputDescription>>& list_inputs,
    PVRush::PVFormat& format,
    PVRush::PVNraw& nraw
)
{
	PVRush::PVSourceCreator_p sc =
	    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("parquet");
	PVRush::PVNrawOutput output(nraw);
	PVRush::PVParquetFileDescription* input_desc = new PVRush::PVParquetFileDescription(QStringList{file});
	list_inputs << PVRush::PVInputDescription_p(input_desc);
	input_desc->disable_multi_inputs(true);
	PVRush::PVParquetAPI parquet_api(input_desc);
	format = PVRush::PVFormat(parquet_api.get_format().documentElement());
	PVRush::PVExtractor extractor(format, output, sc, list_inputs);

	PVRush::PVControllerJob_p job = extractor.process_from_agg_idxes(0, IMPORT_PIPELINE_ROW_COUNT_LIMIT);
	job->wait_end();
}

/**
 * Check every column is mapped to the pvcop type its name is made of.
 */
static void check_format(const PVRush::PVFormat& format)
{
	const QList<PVRush::PVAxisFormat>& axes = format.get_axes();
	const std::shared_ptr<arrow::Schema> schema = test_schema();
	const arrow::FieldVector& fields = schema->fields();

	PV_VALID((size_t)axes.size(), fields.size());

	for (size_t i = 0; i < fields.size(); i++) {
		const QString expected_type = QString::fromStdString(fields[i]->name()).split("#")[0];
		PV_VALID(axes[i].get_type().toStdString(), expected_type.toStdString());
	}
}

static void check_values(const PVRush::PVNraw& nraw)
{
	PV_VALID((size_t)nraw.row_count(), ROW_COUNT);
	PV_VALID((size_t)nraw.column_count(), test_schema()->fields().size());

	const pvcop::db::array& large_string_column = nraw.column(PVCol(0));
	const pvcop::db::array& large_binary_column = nraw.column(PVCol(1));
	const pvcop::db::array& large_list_column = nraw.column(PVCol(2));
	const pvcop::db::array& fixed_size_list_column = nraw.column(PVCol(3));
	const pvcop::db::array& string_view_column = nraw.column(PVCol(4));
	const pvcop::db::array& binary_view_column = nraw.column(PVCol(5));
	const pvcop::db::array& half_float_column = nraw.column(PVCol(6));
	const std::vector<const pvcop::db::array*> decimals = {
		&nraw.column(PVCol(7)), &nraw.column(PVCol(8)), &nraw.column(PVCol(9)), &nraw.column(PVCol(10))
	};
	const std::vector<const decimal_values*> expected_decimals = {
		&decimal32_values, &decimal64_values, &decimal128_values, &decimal256_values
	};
	const std::vector<const pvcop::db::array*> durations = {
		&nraw.column(PVCol(11)), &nraw.column(PVCol(12)), &nraw.column(PVCol(13)), &nraw.column(PVCol(14))
	};
	const std::vector<const duration_values*> expected_durations = {
		&duration_s_values, &duration_ms_values, &duration_us_values, &duration_ns_values
	};

	const pvcop::core::array<float>& float_values = half_float_column.to_core_array<float>();

	for (size_t row = 0; row < ROW_COUNT; row++) {

		const size_t pattern_row = row % PATTERN_SIZE;

		// null values
		for (int col = 0; col < nraw.column_count(); col++) {
			const bool expected_valid = col == FIXED_SIZE_LIST_COLUMN or not is_null_row(row);
			PV_VALID(nraw.column(PVCol(col)).is_valid(row), expected_valid);
		}

		// FIXED_SIZE_LIST (never holds any null value, see above)
		PV_VALID(fixed_size_list_column.at(row), expected_fixed_size_lists[pattern_row]);

		if (is_null_row(row)) {
			continue;
		}

		// LARGE_STRING
		std::string expected_string = *large_strings[pattern_row];
		boost::replace_all(expected_string, "\n", "\\n");
		PV_VALID(large_string_column.at(row), expected_string);

		// STRING_VIEW
		PV_VALID(string_view_column.at(row), *string_views[pattern_row]);

		// LARGE_BINARY and BINARY_VIEW
		PV_VALID(large_binary_column.at(row), expected_checksums[pattern_row]);
		PV_VALID(binary_view_column.at(row), expected_checksums[pattern_row]);

		// LARGE_LIST
		PV_VALID(large_list_column.at(row), expected_large_lists[pattern_row]);

		// HALF_FLOAT
		PV_VALID(float_values[row], *half_floats[pattern_row]);

		// DECIMAL[32|64|128|256]
		for (size_t i = 0; i < decimals.size(); i++) {
			const pvcop::core::array<double>& double_values = decimals[i]->to_core_array<double>();
			PV_VALID(double_values[row], expected_decimals[i]->expected_doubles[pattern_row]);
		}

		// DURATION
		for (size_t i = 0; i < durations.size(); i++) {
			PV_VALID(durations[i]->at(row), expected_durations[i]->expected_strings[pattern_row]);
		}
	}
}

UNICODE_MAIN()
{
	(void) argc;
	(void) argv;

	pvtest::init_ctxt();

	const std::string& parquet_file =
	    PVRush::PVNrawCacheManager::nraw_dir().toStdString() + "/extended_types.parquet";
	generate_parquet_file(parquet_file);

	QList<std::shared_ptr<PVRush::PVInputDescription>> list_inputs;
	PVRush::PVFormat format;
	PVRush::PVNraw nraw;
	import_file(QString::fromStdString(parquet_file), list_inputs, format, nraw);

	check_format(format);
	check_values(nraw);

	// Export back to parquet and reimport, to make sure the types survive a round trip
	const std::string& exported_parquet_file = pvtest::get_tmp_filename() + ".parquet";
	PVCore::PVSelBitField sel(nraw.row_count());
	sel.select_all();
	PVRush::PVParquetExporter exporter(list_inputs, nraw);
	exporter.export_rows(exported_parquet_file, sel);

	QList<std::shared_ptr<PVRush::PVInputDescription>> reimported_list_inputs;
	PVRush::PVFormat reimported_format;
	PVRush::PVNraw reimported_nraw;
	import_file(QString::fromStdString(exported_parquet_file), reimported_list_inputs,
	            reimported_format, reimported_nraw);

	check_format(reimported_format);
	check_values(reimported_nraw);

	// Cleanup files
	std::remove(parquet_file.c_str());
	std::remove(exported_parquet_file.c_str());

	return 0;
}
