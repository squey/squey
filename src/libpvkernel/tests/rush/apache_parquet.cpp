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

#include <memory>

#include "../../plugins/common/parquet/PVParquetAPI.h"
#include "../../plugins/common/parquet/PVParquetFileDescription.h"
#include "../../plugins/input-types/parquet/PVParquetExporter.h"
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVCSVExporter.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/core/squey_assert.h>

#include "common.h"

#include <pvlogger.h>
#include <filesystem>
#include <type_traits>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <arrow/type_fwd.h>
#include <arrow/ipc/api.h>

#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include <QFileInfo>


template <typename T>
T sanitize_value(T val)
{
    return val;
}

template <typename T>
std::string sanitize_value_string(T val)
{
    return std::to_string(val);
}

int64_t sanitize_timestamp_ms(int64_t val)
{
    return val % 100000000000;
}

int64_t sanitize_timestamp_us(int64_t val)
{
    return val % 100000000000 * 1000;
}

int64_t sanitize_timestamp_ns(int64_t val)
{
    return val % 100000000000 * 1000000;
}

int32_t sanitize_date32(int32_t val)
{
    return std::abs(val) % 1000000;
}

int32_t sanitize_time32_sec(int32_t val)
{
    return std::abs(val) % 10000;
}

int32_t sanitize_time32_ms(int32_t val)
{
    return std::abs(val) % 10000 * 1000;
}

int64_t sanitize_time64_us(int64_t val)
{
    return std::abs(val) % 10000000000;
}

int64_t sanitize_time64_ns(int64_t val)
{
    return std::abs(val) % 10000000000 * 1000;
}

template <typename B>
B create_array_builder(const std::shared_ptr<arrow::DataType>&)
{
    return B();
}

template <>
arrow::TimestampBuilder create_array_builder(const std::shared_ptr<arrow::DataType>& type)
{
    return arrow::TimestampBuilder(type, arrow::default_memory_pool());
}

template <>
arrow::Time32Builder create_array_builder(const std::shared_ptr<arrow::DataType>& type)
{
    return arrow::Time32Builder(type, arrow::default_memory_pool());
}

template <>
arrow::Time64Builder create_array_builder(const std::shared_ptr<arrow::DataType>& type)
{
    return arrow::Time64Builder(type, arrow::default_memory_pool());
}

template <>
arrow::FixedSizeBinaryBuilder create_array_builder(const std::shared_ptr<arrow::DataType>& type)
{
    return arrow::FixedSizeBinaryBuilder(type, arrow::default_memory_pool());
}

template <typename B, typename T, typename F>
std::shared_ptr<arrow::Array> generate_array(size_t size, const F& sanitize_value, const std::shared_ptr<arrow::DataType>& type = {})
{
    B builder = create_array_builder<B>(type);
    auto status = builder.Resize(size);

    T min_value = std::numeric_limits<T>::min();
    T max_value = std::numeric_limits<T>::max();
    if constexpr (std::is_same<T, float>::value) {
        min_value = static_cast<float>(std::numeric_limits<uint32_t>::min());
        max_value = static_cast<float>(std::numeric_limits<uint32_t>::max());
    }
    else if constexpr (std::is_same<T, double>::value) {
        min_value = static_cast<double>(std::numeric_limits<uint64_t>::min());
        max_value = static_cast<double>(std::numeric_limits<uint64_t>::max());
    }
    T init_value = max_value;
    __int128_t v = (__int128_t)min_value - (__int128_t)max_value +1;
    size_t decrement = (v < 0 ? -v : v)/(size-2);
    decrement = std::max((size_t)1, decrement);
    status = builder.Append(sanitize_value(max_value));
    for (size_t i = 1; i < size-1; i++) {
        if (i % 7 == 0) {
            status = builder.AppendNull();
        }
        else {
            if constexpr (std::is_same<T, float>::value or std::is_same<T, double>::value) {
                T ratio = static_cast<T>(i) / (size - 1);
                T value = min_value + (max_value - min_value) * ratio;
                status = builder.Append(value);
            }
            else {
                status = builder.Append(sanitize_value(init_value -= decrement));
            }
        }
    }
    status = builder.Append(sanitize_value(min_value+1));

    return builder.Finish().ValueOrDie();
}

std::string generate_parquet_file(std::shared_ptr<arrow::Schema>& schema, const size_t array_size, const std::string fname)
{
    // Create Arrow schema with various column types
    schema = arrow::schema({
        arrow::field("string", arrow::boolean()),
        arrow::field("number_int8", arrow::int8()),
        arrow::field("number_int16", arrow::int16()),
        arrow::field("number_int32", arrow::int32()),
        arrow::field("number_int64", arrow::int64()),
        arrow::field("number_uint8", arrow::uint8()),
        arrow::field("number_uint16", arrow::uint16()),
        arrow::field("number_uint32", arrow::uint32()),
        arrow::field("number_uint64", arrow::uint64()),
        arrow::field("number_float", arrow::float32()),
        arrow::field("number_double", arrow::float64()),
        arrow::field("string#1", arrow::utf8()),
        arrow::field("string#2", arrow::fixed_size_binary(4)),
        arrow::field("string#3", arrow::binary()),
        arrow::field("time", arrow::timestamp(arrow::TimeUnit::MILLI)),
        arrow::field("time#1", arrow::timestamp(arrow::TimeUnit::MICRO)),
        arrow::field("time#2", arrow::timestamp(arrow::TimeUnit::NANO)),
        arrow::field("time#3", arrow::date32()),
        arrow::field("duration", arrow::time32(arrow::TimeUnit::SECOND)),
        arrow::field("duration#1", arrow::time32(arrow::TimeUnit::MILLI)),
        arrow::field("duration#2", arrow::time64(arrow::TimeUnit::MICRO)),
        arrow::field("duration#3", arrow::time64(arrow::TimeUnit::NANO)),
        arrow::field("structure", arrow::struct_({
        arrow::field("number_uint8#1", arrow::uint8()),
        arrow::field("string#4", arrow::utf8())
        }))
    });

    auto bool_array         = generate_array<arrow::BooleanBuilder, bool>(array_size, sanitize_value<bool>);
    auto int8_array         = generate_array<arrow::Int8Builder, int8_t>(array_size, sanitize_value<int8_t>);
    auto int16_array        = generate_array<arrow::Int16Builder, int16_t>(array_size, sanitize_value<int16_t>);
    auto int32_array        = generate_array<arrow::Int32Builder, int32_t>(array_size, sanitize_value<int32_t>);
    auto int64_array        = generate_array<arrow::Int64Builder, int64_t>(array_size, sanitize_value<int64_t>);
    auto uint8_array        = generate_array<arrow::UInt8Builder, uint8_t>(array_size, sanitize_value<uint8_t>);
    auto uint16_array       = generate_array<arrow::UInt16Builder, uint16_t>(array_size, sanitize_value<uint16_t>);
    auto uint32_array       = generate_array<arrow::UInt32Builder, uint32_t>(array_size, sanitize_value<uint32_t>);
    auto uint64_array       = generate_array<arrow::UInt64Builder, uint64_t>(array_size, sanitize_value<uint64_t>);
    auto float_array        = generate_array<arrow::FloatBuilder, float>(array_size, sanitize_value<float>);
    auto double_array       = generate_array<arrow::DoubleBuilder, double>(array_size, sanitize_value<double>);
    auto string_array       = generate_array<arrow::StringBuilder, int8_t>(array_size, sanitize_value_string<int8_t>);
    auto fixed_binary_array = generate_array<arrow::FixedSizeBinaryBuilder, uint64_t>(array_size, sanitize_value_string<uint64_t>, arrow::fixed_size_binary(4));
    auto binary_array       = generate_array<arrow::BinaryBuilder, uint64_t>(array_size, sanitize_value_string<uint64_t>);
    auto timestamp_ms_array = generate_array<arrow::TimestampBuilder, int64_t>(array_size, sanitize_timestamp_ms, arrow::timestamp(arrow::TimeUnit::MILLI));
    auto timestamp_us_array = generate_array<arrow::TimestampBuilder, int64_t>(array_size, sanitize_timestamp_us, arrow::timestamp(arrow::TimeUnit::MICRO));
    auto timestamp_ns_array = generate_array<arrow::TimestampBuilder, int64_t>(array_size, sanitize_timestamp_ns, arrow::timestamp(arrow::TimeUnit::NANO));
    auto date32_array       = generate_array<arrow::Date32Builder, int32_t>(array_size, sanitize_date32);
    auto time32_sec_array   = generate_array<arrow::Time32Builder, int32_t>(array_size, sanitize_time32_sec, arrow::time32(arrow::TimeUnit::SECOND));
    auto time32_ms_array    = generate_array<arrow::Time32Builder, int32_t>(array_size, sanitize_time32_ms, arrow::time32(arrow::TimeUnit::MILLI));
    auto time64_us_array    = generate_array<arrow::Time64Builder, int64_t>(array_size, sanitize_time64_us, arrow::time64(arrow::TimeUnit::MICRO));
    auto time64_ns_array    = generate_array<arrow::Time64Builder, int64_t>(array_size, sanitize_time64_ns, arrow::time64(arrow::TimeUnit::NANO));
    auto struct_array       = arrow::StructArray::Make({uint8_array, string_array}, {arrow::field("", arrow::uint8()), arrow::field("", arrow::utf8())}).ValueOrDie();

    // Create Arrow RecordBatch with sample data
    arrow::RecordBatchVector batches;
    std::shared_ptr<arrow::RecordBatch> record_batch = arrow::RecordBatch::Make(schema, array_size, {
        bool_array,
        int8_array,
        int16_array,
        int32_array,
        int64_array,
        uint8_array,
        uint16_array,
        uint32_array,
        uint64_array,
        float_array,
        double_array,
        string_array,
        fixed_binary_array,
        binary_array,
        timestamp_ms_array,
        timestamp_us_array,
        timestamp_ns_array,
        date32_array,
        time32_sec_array,
        time32_ms_array,
        time64_us_array,
        time64_ns_array,
        struct_array
    });

    batches.push_back(record_batch);

    // Write Arrow RecordBatch to Parquet file
    std::string test_file = PVRush::PVNrawCacheManager::nraw_dir().toStdString() + "/" + fname;
    auto output_file = arrow::io::FileOutputStream::Open(test_file).ValueOrDie();
    auto file_writer = parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), output_file).ValueOrDie();
    std::shared_ptr<arrow::Table> table = arrow::Table::FromRecordBatches({record_batch}).ValueOrDie();
    PARQUET_THROW_NOT_OK(file_writer->WriteTable(*table));
    PARQUET_THROW_NOT_OK(file_writer->Close());

    return test_file;
}

void import_files(
    const QStringList& files,
    const std::shared_ptr<arrow::Schema>& schema,
    QList<std::shared_ptr<PVRush::PVInputDescription>>& list_inputs,
    PVRush::PVFormat& format,
    PVRush::PVNraw& nraw
)
{
    PVRush::PVSourceCreator_p sc =
        LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("parquet");
    PVRush::PVNrawOutput output(nraw);
    PVRush::PVParquetFileDescription* input_desc = new PVRush::PVParquetFileDescription(files);
    list_inputs << PVRush::PVInputDescription_p(input_desc);
    input_desc->disable_multi_inputs(true);
    PVRush::PVParquetAPI parquet_api(input_desc);
    format = PVRush::PVFormat(parquet_api.get_format().documentElement());
    PVRush::PVExtractor extractor(format, output, sc, list_inputs);

    // Import data
    auto start = std::chrono::system_clock::now();
    PVRush::PVControllerJob_p job =
        extractor.process_from_agg_idxes(0, IMPORT_PIPELINE_ROW_COUNT_LIMIT);
    job->wait_end();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << diff.count();

    // Check proper types are used in PVFormat
    const auto& axes_format = format.get_axes();
    std::shared_ptr<arrow::Schema> flattened_schema = PVRush::PVParquetAPI::flatten_schema(schema);
    const auto& schema_fields = flattened_schema->fields();
    PV_ASSERT_VALID(std::equal(axes_format.begin(), axes_format.end(), schema_fields.begin(), [](const auto& axis_format, const auto& field) {
        QString field_name = QString::fromStdString(field->name());
        int dot_index = field_name.indexOf('.');
        auto type_name = field_name.mid(dot_index + 1).split("#")[0];
        return axis_format.get_type() == type_name;
    }));

    // Check null values are properly set
#pragma omp parallel for schedule(dynamic)
    for (int col = 0; col < nraw.column_count(); col++) {
        const pvcop::db::array& column = nraw.column(PVCol(col));
        PV_VALID(column.is_valid(0), true);
        for (size_t j = 1; j < column.size()/2-1; j++) {
            PV_VALID(column.is_valid(j), (j % 7 != 0));
        }
        PV_VALID(column.is_valid(column.size()/2-1), true);
        PV_VALID(column.is_valid(column.size()/2), true);
        for (size_t j = 1; j < column.size()/2-1; j++) {
            size_t j2 = j + column.size()/2;
            PV_VALID(column.is_valid(j2), (j % 7 != 0));
        }
        PV_VALID(column.is_valid(column.size()-1), true);
    }
}

UNICODE_MAIN()
{
	if (argc <= 4) {
		std::cerr
		    << "Usage:"
		    << " <csv_ref1 csv_ref2 parquet_file csv_ref3>"
		    << std::endl;
		return 1;
	}
    pvtest::init_ctxt();

#ifdef _WIN32
    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
    std::string parquet_test_file = conv.to_bytes(argv[3]);
    std::string csv_ref_file = conv.to_bytes(argv[4]);
#else
    std::string parquet_test_file = argv[3];
    std::string csv_ref_file = argv[4];
#endif

    std::vector<size_t> sizes = { 150000, 65544 };
    //std::vector<size_t> sizes = { 128 };
    for (size_t i = 0; i < sizes.size(); i++) {
#ifdef _WIN32
        std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
        const std::string csv_ref = conv.to_bytes(argv[i+1]);
#else
        const std::string& csv_ref = argv[i+1];
#endif

        std::shared_ptr<arrow::Schema> schema;
        std::string test_file = generate_parquet_file(schema, sizes[i], QFileInfo(QString::fromStdString(csv_ref)).fileName().toStdString() + ".parquet");

        // Import multiple parquet files
        QStringList files;
        files << QString::fromStdString(test_file);
        files << QString::fromStdString(test_file);
        QList<std::shared_ptr<PVRush::PVInputDescription>> list_inputs;
        PVRush::PVFormat format;
        PVRush::PVNraw nraw;
        import_files(files, schema, list_inputs, format, nraw);

        // Export selected lines as CSV file and check if content is same as a csv reference file
        PVCore::PVSelBitField sel(nraw.row_count());
        sel.select_odd();
        const std::string& output_csv_file = pvtest::get_tmp_filename() + ".parquet.csv";
        PVRush::PVCSVExporter::export_func_f export_func =
            [&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
                const std::string& quote) { return nraw.export_line(row, cols, sep, quote); };
        PVRush::PVCSVExporter exp_csv(format.get_axes_comb(), nraw.row_count(), export_func);
        exp_csv.export_rows(output_csv_file, sel);
        std::cout << std::endl << output_csv_file << " - " << csv_ref << std::endl;
        PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(output_csv_file, csv_ref));

        // Export selected lines as parquet file
        const std::string& output_parquet_file = pvtest::get_tmp_filename() + ".parquet";
        PVRush::PVParquetExporter exp_parquet(list_inputs, nraw);
        exp_parquet.export_rows(output_parquet_file, sel);

        // Reload exported parquet file
        QStringList files2;
        files2 << QString::fromStdString(output_parquet_file);
        QList<std::shared_ptr<PVRush::PVInputDescription>> list_inputs2;
        PVRush::PVFormat format2;
        PVRush::PVNraw nraw2;
        import_files(files2, schema, list_inputs2, format2, nraw2);

        // Export reloaded parquet file as CSV and check if content is same as the original csv reference file
        PVCore::PVSelBitField sel2(nraw2.row_count());
        sel2.select_all();
        const std::string& output_csv_file2 = pvtest::get_tmp_filename() + ".parquet.csv";
        PVRush::PVCSVExporter::export_func_f export_func2 =
            [&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
                const std::string& quote) { return nraw2.export_line(row, cols, sep, quote); };
        PVRush::PVCSVExporter exp_csv2(format.get_axes_comb(), nraw2.row_count(), export_func2);
        exp_csv2.export_rows(output_csv_file2, sel2);
        std::cout << std::endl << output_csv_file2 << " - " << csv_ref << std::endl;
        PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(output_csv_file2, csv_ref));

        // Cleanup files
        std::remove(output_csv_file.c_str());
        std::remove(output_csv_file2.c_str());
        std::remove(output_parquet_file.c_str());
    }

    // Import multiple parquet files
    QStringList files;
    files << QString::fromStdString(parquet_test_file);
    files << QString::fromStdString(parquet_test_file);
    QList<std::shared_ptr<PVRush::PVInputDescription>> list_inputs;
    PVRush::PVFormat format;
    PVRush::PVNraw nraw;

    PVRush::PVSourceCreator_p sc =
        LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("parquet");
    PVRush::PVNrawOutput output(nraw);
    PVRush::PVParquetFileDescription* input_desc = new PVRush::PVParquetFileDescription(files);
    list_inputs << PVRush::PVInputDescription_p(input_desc);
    input_desc->disable_multi_inputs(true);
    PVRush::PVParquetAPI parquet_api(input_desc);
    format = PVRush::PVFormat(parquet_api.get_format().documentElement());
    PVRush::PVExtractor extractor(format, output, sc, list_inputs);

    // Import data
    auto start = std::chrono::system_clock::now();
    PVRush::PVControllerJob_p job =
        extractor.process_from_agg_idxes(0, IMPORT_PIPELINE_ROW_COUNT_LIMIT);
    job->wait_end();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << diff.count();

    // Export selected lines as CSV file and check if content is same as a csv reference file
    PVCore::PVSelBitField sel(nraw.row_count());
    sel.select_none();
    for (size_t i : {0, 2, 4, 6, 8, 11, 13, 15, 17, 19}) {
        sel.set_line(i, true);
    }
    const std::string& output_csv_file = pvtest::get_tmp_filename() + ".parquet.csv";
    PVRush::PVCSVExporter::export_func_f export_func =
        [&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
            const std::string& quote) { return nraw.export_line(row, cols, sep, quote); };
    PVRush::PVCSVExporter exp_csv(format.get_axes_comb(), nraw.row_count(), export_func);
    exp_csv.export_rows(output_csv_file, sel);
    std::cout << std::endl << output_csv_file << " - " << csv_ref_file << std::endl << std::flush;
    PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(output_csv_file, csv_ref_file));

	return 0;
}
